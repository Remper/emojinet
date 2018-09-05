package eu.fbk.fm.emojinet;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import eu.fbk.fm.alignments.twitter.TwitterDeserializer;
import eu.fbk.fm.alignments.utils.flink.JsonObjectProcessor;
import eu.fbk.utils.core.CommandLine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

/**
 * DESC
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class MergeDistributions implements JsonObjectProcessor {
    private static final Logger LOGGER = LoggerFactory.getLogger(GatherUserContext.class);
    private static final Gson GSON = TwitterDeserializer.getDefault().getBuilder().create();

    private static final String INPUT_PATH = "input";

    private static CommandLine.Parser provideParameterList() {
        return CommandLine.parser()
                .withOption("i", INPUT_PATH,
                        "Emoji distributions", "FILE",
                        CommandLine.Type.STRING, true, false, true);
    }

    public void start(Path input) throws IOException {
        HashMap<Long, HashMap<String, Integer>> distribution = new HashMap<>();
        AtomicInteger coincidingUsers = new AtomicInteger();
        Files.list(input)
            .filter(file -> file.toString().endsWith(".tsv"))
            .flatMap(file -> {
                try {
                    return Files.lines(file);
                } catch (IOException e) {
                    LOGGER.warn("Error while opening: "+file.toString(), e);
                    return Stream.empty();
                }
            }).forEach(line -> {
                if (line.startsWith("uid\temoji")) {
                    return;
                }
                String[] record = line.split("\t");
                long id = Long.valueOf(record[0]);

                Type type = new TypeToken<Map<String, Integer>>(){}.getType();
                Map<String, Integer> userDistribution = GSON.fromJson(record[1], type);

                if (!distribution.containsKey(id)) {
                    distribution.put(id, new HashMap<>());
                } else {
                    coincidingUsers.getAndIncrement();
                }
                HashMap<String, Integer> finalDistribution = distribution.get(id);
                for (Map.Entry<String, Integer> entry : userDistribution.entrySet()) {
                    finalDistribution.put(entry.getKey(), finalDistribution.getOrDefault(entry.getKey(), 0)+entry.getValue());
                }
            });

        FileWriter writer = new FileWriter(input.resolve("merged_dist.tsv").toFile());
        writer.write("uid\temoji");
        for (Map.Entry<Long, HashMap<String, Integer>> user : distribution.entrySet()) {
            try {
                writer.write('\n');
                writer.write(String.valueOf(user.getKey()));
                writer.write('\t');
                writer.write(GSON.toJson(user.getValue()));
            } catch (IOException e) {
                LOGGER.error("Error while writing distribution for user "+user.getKey(), e);
            }
        }
        writer.close();
        LOGGER.info("Done. Written "+distribution.size()+" users. Coinciding users: "+coincidingUsers.get());
    }

    public static void main(String[] args) throws Exception {
        MergeDistributions extractor = new MergeDistributions();

        try {
            // Parse command line
            final CommandLine cmd = provideParameterList().parse(args);

            final String distPath = cmd.getOptionValue(INPUT_PATH, String.class);

            //noinspection ConstantConditions
            extractor.start(Paths.get(distPath));
        } catch (final Throwable ex) {
            // Handle exception
            CommandLine.fail(ex);
        }
    }
}
