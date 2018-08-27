package com.company;

import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
import eu.fbk.dh.tint.runner.TintPipeline;
import eu.fbk.dh.tint.runner.TintRunner;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class Main {

    public static void main(String[] args) throws IOException {


        TintPipeline pipeline = new TintPipeline();

        pipeline.loadDefaultProperties();
        pipeline.load();

        JsonParser parser = new JsonParser();

        JsonReader jr = new JsonReader(new FileReader("evalita_train.json"));

        jr.setLenient(true);

        Integer i = 1;
        Integer missed = 0;

        File file_mid = new File("annotation.json");
        FileOutputStream fop = new FileOutputStream(file_mid);

        while(jr.hasNext()){
            if(i % 100 == 0) {
                System.out.println(i);
            }
            i++;
            String text = new String();
            Object obj = parser.parse(jr);
            JSONObject jo = null;
            try {
                jo = new JSONObject(obj.toString());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            try {
                text = jo.get("text_no_emoji").toString();
            } catch (JSONException e) {
                e.printStackTrace();
            }
            InputStream stream = new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8));
            try {
                pipeline.run(stream, fop, TintRunner.OutputFormat.JSON);
            } catch(Exception e) {
                missed++;
            }

        }

        fop.close();

        System.out.println("Missed " + missed.toString());

    }
}



