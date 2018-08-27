package eu.fbk.fm.emojinet;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import eu.fbk.fm.alignments.twitter.SearchRunner;
import eu.fbk.fm.alignments.twitter.TwitterCredentials;
import eu.fbk.fm.alignments.twitter.TwitterDeserializer;
import eu.fbk.fm.alignments.twitter.TwitterService;
import eu.fbk.fm.alignments.utils.flink.JsonObjectProcessor;
import eu.fbk.fm.vectorize.preprocessing.text.TextExtractor;
import eu.fbk.utils.core.CommandLine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import twitter4j.Status;
import twitter4j.TwitterObjectFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Gathers tweet objects for all tweet ids
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class GatherTweets implements JsonObjectProcessor {
    private static final Logger LOGGER = LoggerFactory.getLogger(GatherTweets.class);
    private static final Gson GSON = TwitterDeserializer.getDefault().getBuilder().create();

    private static final String INPUT_PATH = "input";
    private static final String OUTPUT_PATH = "output";
    private static final String CREDENTIALS_PATH = "credentials";

    public void start(Path input, Path output, TwitterService service) throws IOException {
        List<ResolveResult> tweets = Files.lines(input)
            .parallel()
            .map(line -> GSON.fromJson(line, JsonObject.class))
            .map(obj -> ResolveResult.create(Long.valueOf(get(obj, String.class, "tid"))))
            .collect(Collectors.toList());

        LOGGER.info(String.format("Extracted %d tweets", tweets.size()));

        AtomicInteger processed = new AtomicInteger();
        AtomicInteger lost = new AtomicInteger();
        Function<List<ResolveResult>, Integer> resolveStatuses = (ids -> {
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            boolean retry = true;
            while (retry) {
                try {
                    HashMap<Long, ResolveResult> mapping = new HashMap<>();
                    long[] rawIds = new long[ids.size()];
                    int pointer = 0;
                    for (ResolveResult id : ids) {
                        mapping.put(id.id, id);
                        rawIds[pointer] = id.id;
                        pointer++;
                    }
                    List<Status> statuses = service.lookupStatuses(rawIds);
                    if (ids.size() != statuses.size()) {
                        lost.addAndGet(ids.size() - statuses.size());
                    }
                    for (Status status : statuses) {
                        mapping.get(status.getId()).object = GSON.fromJson(TwitterObjectFactory.getRawJSON(status), JsonObject.class);
                    }

                    retry = false;
                } catch (TwitterService.RateLimitException limit) {
                    try {
                        LOGGER.info(String.format("Limit reached, sleeping for %d seconds", limit.remaining));
                        Thread.sleep(limit.remaining * 1000);
                    } catch (InterruptedException e) {
                    }
                }
            }

            return processed.incrementAndGet();
        });

        List<Callable<Integer>> blocks = new LinkedList<>();
        int curOffset = 0;
        while (curOffset < tweets.size()) {
            final List<ResolveResult> ids = tweets.subList(curOffset, curOffset+100);
            blocks.add(() -> resolveStatuses.apply(ids));
            curOffset += 100;
        }

        ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1);
        ScheduledFuture<?> progressCheck = executor.scheduleWithFixedDelay(() -> {
            LOGGER.info(String.format("  Progress: %5d/%5d (lost tweets: %d)", processed.get(), blocks.size(), lost.get()));
        }, 1, 30, TimeUnit.SECONDS);

        ForkJoinPool forkJoinPool = new ForkJoinPool();
        forkJoinPool.invokeAll(blocks);
        forkJoinPool.shutdown();
        progressCheck.cancel(true);
        executor.shutdownNow();

        TextExtractor extractor = new TextExtractor(true);
        try (FileWriter writer = new FileWriter(output.toFile())) {
            int flost = 0;
            int written = 0;
            for (ResolveResult result : tweets) {
                if (result.object == null) {
                    flost++;
                    continue;
                }

                if (written > 0) {
                    writer.write('\n');
                }
                writer.write(GSON.toJson(result.object).replaceAll("\\n|\\s|\\t", " "));
                writer.write('\t');
                writer.write(extractor.map(result.object));
                written++;
            }
            LOGGER.info(String.format("Written %d, lost %d (%.2f%%)", written, flost, ((float)flost / (written+flost)) * 100));
        }
    }

    private static class ResolveResult {
        long id;
        JsonObject object = null;

        public static ResolveResult create(long id) {
            ResolveResult result = new ResolveResult();
            result.id = id;
            return result;
        }
    }

    private static CommandLine.Parser provideParameterList() {
        return CommandLine.parser()
                .withOption("i", INPUT_PATH,
                        "EVALITA dataset", "FILE",
                        CommandLine.Type.STRING, true, false, true)
                .withOption("o", OUTPUT_PATH,
                        "Expanded EVALITA dataset", "FILE",
                        CommandLine.Type.STRING, true, false, true)
                .withOption("c", CREDENTIALS_PATH,
                        "File with twitter credentials", "FILE",
                        CommandLine.Type.STRING, true, false, true);
    }

    public static void main(String[] args) throws Exception {
        GatherTweets extractor = new GatherTweets();

        try {
            // Parse command line
            final CommandLine cmd = provideParameterList().parse(args);

            final String tweetsPath = cmd.getOptionValue(INPUT_PATH, String.class);
            final String outputPath = cmd.getOptionValue(OUTPUT_PATH, String.class);
            final String credentialsPath = cmd.getOptionValue(CREDENTIALS_PATH, String.class);

            //noinspection ConstantConditions
            TwitterCredentials[] credentials = TwitterCredentials.credentialsFromFile(new File(credentialsPath));
            TwitterService service = new TwitterService(SearchRunner.createInstances(credentials));

            //noinspection ConstantConditions
            extractor.start(Paths.get(tweetsPath), Paths.get(outputPath), service);
        } catch (final Throwable ex) {
            // Handle exception
            CommandLine.fail(ex);
        }
    }
}
