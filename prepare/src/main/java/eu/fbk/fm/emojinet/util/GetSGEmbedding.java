package eu.fbk.fm.emojinet.util;

import com.google.gson.*;
import eu.fbk.fm.alignments.utils.flink.JsonObjectProcessor;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.URI;
import java.net.URISyntaxException;

/**
 * Provides social graph embedding from the embedding API
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class GetSGEmbedding implements JsonObjectProcessor {

    protected static final Logger logger = LoggerFactory.getLogger(eu.fbk.fm.alignments.scorer.embeddings.EmbeddingsProvider.class.getName());

    protected final String embName;
    private final String host = "localhost";
    private final int port = 5241;
    private final CloseableHttpClient client = HttpClients.createDefault();

    private URI url;

    public GetSGEmbedding(String embName) throws URISyntaxException {
        if (embName.length() == 0) {
            throw new IllegalArgumentException("embName can't be empty");
        }

        this.embName = embName;
        init();
    }

    public String getSubspaceId() {
        return "emb_" + this.embName + "_w";
    }

    private void init() throws URISyntaxException {
        url = new URIBuilder().setScheme("http").setHost(host).setPort(port).setPath("/transform/"+embName).build();
    }

    public double[] predict(Serializable[] features, Float[] weights) {
        Gson gson = new GsonBuilder().create();
        double[] result = null;
        CloseableHttpResponse response = null;
        try {
            URIBuilder requestBuilder = new URIBuilder(url).setParameter("followees", gson.toJson(features));
            if (weights != null) {
                requestBuilder.setParameter("weights", gson.toJson(weights));
            }
            URI requestURI = requestBuilder.build();

            response = client.execute(new HttpGet(requestURI));
            if (response.getStatusLine().getStatusCode() >= 400) {
                throw new IOException(String.format(
                        "Embeddings endpoint didn't understand the request. Code: %d",
                        response.getStatusLine().getStatusCode()
                ));
            }
            JsonObject object = gson.fromJson(new InputStreamReader(response.getEntity().getContent()), JsonObject.class);
            JsonArray results = get(object, JsonArray.class, "data", "embedding");
            if (results == null) {
                throw new IOException("Incorrect response: "+gson.toJson(object));
            }
            result = new double[results.size()];
            int pointer = 0;
            for (JsonElement ele : results) {
                result[pointer] = ele.getAsDouble();
                pointer++;
            }
        } catch (URISyntaxException | IOException | JsonSyntaxException e) {
            e.printStackTrace();
            logger.error(String.format(
                    "Error in subspace %s while requesting for followees %s and weights %s",
                    getSubspaceId(),
                    gson.toJson(features), gson.toJson(weights)
            ), e);
        }

        //Checking if everything is properly closed
        if (response != null) {
            try {
                response.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        //Returning empty result in case of an error
        if (result == null) {
            return new double[0];
        }
        return result;
    }
}
