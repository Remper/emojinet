package eu.fbk.fm.emojinet;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import eu.fbk.fm.alignments.twitter.SearchRunner;
import eu.fbk.fm.alignments.twitter.TwitterCredentials;
import eu.fbk.fm.alignments.twitter.TwitterDeserializer;
import eu.fbk.fm.alignments.twitter.TwitterService;
import eu.fbk.fm.alignments.utils.flink.JsonObjectProcessor;
import eu.fbk.fm.emojinet.util.GetSGEmbedding;
import eu.fbk.fm.vectorize.preprocessing.text.TextExtractor;
import eu.fbk.utils.core.CommandLine;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import twitter4j.Status;
import twitter4j.TwitterObjectFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Gathers user-related data from the list of UIDs: last 200 tweets
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class GatherUserContext implements JsonObjectProcessor {
    private static final Logger LOGGER = LoggerFactory.getLogger(GatherUserContext.class);
    private static final Gson GSON = TwitterDeserializer.getDefault().getBuilder().create();

    private static final String INPUT_PATH = "input";
    private static final String OUTPUT_PATH = "output";
    private static final String CREDENTIALS_PATH = "credentials";

    public void start(Path input, Path output, TwitterService service) throws IOException {
        Set<Long> users = Files.lines(input)
                .parallel()
                .map(line -> GSON.fromJson(line, JsonObject.class))
                .map(obj -> Long.valueOf(get(obj, String.class, "uid")))
                .collect(Collectors.toSet());

        LOGGER.info(String.format("Extracted %d users", users.size()));

        AtomicInteger processed = new AtomicInteger();
        AtomicInteger statusesCount = new AtomicInteger();
        HashMap<Long, List<JsonObject>> statuses = new HashMap<>();
        Function<Long, Integer> resolveStatuses = (id -> {
            boolean retry = true;
            while (retry) {
                try {

                    List<Status> raw_result = service.getStatuses(id);
                    List<JsonObject> result = raw_result
                            .stream()
                            .map(status -> {
                                JsonObject json = GSON.fromJson(TwitterObjectFactory.getRawJSON(status), JsonObject.class);
                                if (json == null) {
                                    LOGGER.error("Null object for status: "+status.getId()+" "+status.getText());
                                }
                                return json;
                            })
                            .filter(Objects::nonNull)
                            .collect(Collectors.toList());

                    statusesCount.addAndGet(result.size());
                    synchronized (GSON) {
                        statuses.put(id, result);
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

        List<Callable<Integer>> blocks = users
                .stream()
                .map(uid -> (Callable<Integer>)(() -> resolveStatuses.apply(uid)))
                .collect(Collectors.toList());

        AtomicInteger completedBlocks = new AtomicInteger();
        ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1);
        ScheduledFuture<?> progressCheck = executor.scheduleWithFixedDelay(() -> {
            LOGGER.info(String.format(
                "  Progress: %5d/%5d (statuses per user: %.2f, completed blocks: %d)",
                processed.get(), users.size(), (float)statusesCount.get() / processed.get(),
                completedBlocks.get()
            ));
        }, 1, 30, TimeUnit.SECONDS);

        int batch_size = 1000;
        int pointer = 0;
        FileWriter writer = new FileWriter(output.resolve("emoji_dist.tsv").toFile());
        writer.write("uid\temoji");
        while (pointer+batch_size < blocks.size())  {
            ForkJoinPool forkJoinPool = new ForkJoinPool();
            forkJoinPool.invokeAll(blocks.subList(pointer, pointer+batch_size));
            forkJoinPool.shutdown();
            pointer += batch_size;
            completedBlocks.getAndIncrement();

            dump(statuses, writer, output);
        }
        ForkJoinPool forkJoinPool = new ForkJoinPool();
        forkJoinPool.invokeAll(blocks.subList(pointer, blocks.size()));
        forkJoinPool.shutdown();
        dump(statuses, writer, output);
        completedBlocks.getAndIncrement();

        progressCheck.cancel(true);
        executor.shutdownNow();
        writer.close();
    }

    public void startFromResults(Path output) throws IOException, URISyntaxException {
        // Reading .json ending files one by one and writing the resulting distribution
        FileWriter writer = new FileWriter(output.resolve("emoji_dist.tsv").toFile());
        writer.write("uid\tsgemb\temoji");
        AtomicInteger processed = new AtomicInteger();
        GetSGEmbedding embeddingProvider = new GetSGEmbedding("sg300");
        Files.list(output)
            .filter(file -> file.toString().endsWith(".json"))
            .parallel()
            .map(file -> {
                double[] socialEmbedding = new double[0];
                String filename = file.getFileName().toString();
                Long id = Long.valueOf(filename.substring(0, filename.length()-5));
                List<String> emojis = new LinkedList<>();
                try {
                    List<JsonObject> tweets = Files
                        .lines(file)
                        .map(json -> GSON.fromJson(json, JsonObject.class))
                        .collect(Collectors.toList());
                    emojis = tweets.stream()
                        .flatMap(status -> resolveEmoji(status).stream())
                        .collect(Collectors.toList());

                    //Constructing a fake social graph
                    HashMap<Long, Float> socialGraph =  new HashMap<>();
                    float norm = 0.0f;
                    for (JsonObject status : tweets) {
                        JsonObject entities = status.getAsJsonObject("entities");

                        //Retweet author
                        Long originalAuthorId = get(status, Long.class, "retweeted_status", "user", "id");
                        JsonObject retweetedObject = status.getAsJsonObject("retweeted_status");
                        if (originalAuthorId != null) {
                            socialGraph.put(originalAuthorId, socialGraph.getOrDefault(originalAuthorId, 0.0f)+1.0f);
                            norm += 1.0f;
                        }

                        //Mentions
                        JsonArray rawMentions = entities.getAsJsonArray("user_mentions");
                        for (JsonElement rawMention : rawMentions) {
                            Long mentionId = get(rawMention, Long.class, "id");
                            if (mentionId == null) {
                                continue;
                            }
                            socialGraph.put(mentionId, socialGraph.getOrDefault(mentionId, 0.0f)+1.0f);
                            norm += 1.0f;
                        }
                    }
                    Long[] ids = new Long[socialGraph.size()];
                    Float[] weights = new Float[socialGraph.size()];
                    int pointer = 0;
                    for (Map.Entry<Long, Float> entry : socialGraph.entrySet()) {
                        ids[pointer] = entry.getKey();
                        weights[pointer] = entry.getValue() / norm;
                        pointer++;
                    }
                    socialEmbedding = embeddingProvider.predict(ids, weights);
                } catch (IOException e) {
                    LOGGER.error("Error while reading data for user "+id, e);
                }
                return new Tuple3<>(id, socialEmbedding, emojis);
            })
            .forEach(tuple -> {
                synchronized (writer) {
                    dump(tuple.f0, tuple.f2, tuple.f1, writer);
                }
                int proc = processed.incrementAndGet();
                if (proc % 10000 == 0) {
                    LOGGER.info("Processed "+proc+" users");
                }
            });
        writer.close();
        LOGGER.info("Done. Written "+processed.get()+" users.");
    }

    private void dump(long id, List<String> emojis, double[] socialEmbedding, FileWriter writer) {
        HashMap<String, Integer> distribution = new HashMap<>();
        for (String emoji : emojis) {
            distribution.put(emoji, distribution.getOrDefault(emoji, 0)+1);
        }
        try {
            writer.write('\n');
            writer.write(String.valueOf(id));
            writer.write('\t');
            writer.write(GSON.toJson(socialEmbedding));
            writer.write('\t');
            writer.write(GSON.toJson(distribution));
        } catch (IOException e) {
            LOGGER.error("Error while writing distribution for user "+id, e);
        }
    }

    private void dump(HashMap<Long, List<JsonObject>> statuses, FileWriter writer, Path output) throws IOException {
        synchronized (GSON) {
            for (Map.Entry<Long, List<JsonObject>> entry : statuses.entrySet()) {
                List<String> emojis = entry.getValue()
                        .stream()
                        .flatMap(status -> resolveEmoji(status).stream())
                        .collect(Collectors.toList());
                dump(entry.getKey(), emojis, new double[0], writer);
                try (FileWriter userWriter = new FileWriter(output.resolve(entry.getKey().toString()+".json").toFile())) {
                    boolean first = true;
                    for (JsonObject status : entry.getValue()) {
                        if (!first) {
                            userWriter.write('\n');
                        }
                        first = false;
                        GSON.toJson(status, userWriter);
                    }
                }
            }
            LOGGER.info(String.format("Written %d users", statuses.size()));
            statuses.clear();
        }
    }

    private LinkedList<String> resolveEmoji(JsonObject status) {
        String originalText = get(status, String.class, "text");
        if (originalText == null) {
            originalText = get(status, String.class, "full_text");
        }

        LinkedList<String> result = new LinkedList<>();
        originalText.codePoints().forEach(value -> {
            if ((value >= 0x1F600 && value <= 0x1F64F) // Emoticons
                    || (value >= 0x1F900 && value <= 0x1F9FF) // Supplemental Symbols and Pictograms
                    || (value >= 0x2600 && value <= 0x26FF) // Miscellaneous Symbols
                    || (value >= 0x2700 && value <= 0x27BF) // Dingbats
                    || (value >= 0x1F300 && value <= 0x1F5FF)) // Miscellaneous Symbols And Pictographs (Emoji)
            {
                result.add(new String(new int[]{value}, 0, 1));
            }
        });
        return result;
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
                        CommandLine.Type.STRING, true, false, false);
    }

    public static void main(String[] args) throws Exception {
        GatherUserContext extractor = new GatherUserContext();

        try {
            // Parse command line
            final CommandLine cmd = provideParameterList().parse(args);

            final String tweetsPath = cmd.getOptionValue(INPUT_PATH, String.class);
            final String outputPath = cmd.getOptionValue(OUTPUT_PATH, String.class);

            if (!cmd.hasOption(CREDENTIALS_PATH)) {
                //noinspection ConstantConditions
                extractor.startFromResults(Paths.get(outputPath));
                return;
            }

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
