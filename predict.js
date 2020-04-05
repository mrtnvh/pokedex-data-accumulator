const tf = require("@tensorflow/tfjs");
const pokemon = require("./pokemon.json");
const path = require("path");
const pEachSeries = require("p-each-series");
const image = require("./lib/image");
const { HEADER, MODEL_DIR } = require("./constants");

const buildPretrainedModel = async () => {
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
  );

  const layer = mobilenet.getLayer("conv_pw_13_relu");
  return tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output,
  });
};

const makePrediction = (pretrainedModel, model, imagePath, expectedLabel) =>
  image
    .loadSingle(imagePath)
    .then((loadedImage) => {
      const croppedImage = image.crop(loadedImage);
      const resizedImage = image.resize(croppedImage);
      return image.batch(resizedImage);
    })
    .then((loadedImage) => {
      const activatedImage = pretrainedModel.predict(loadedImage);
      loadedImage.dispose();
      return activatedImage;
    })
    .then((activatedImage) => {
      const prediction = model.predict(activatedImage);
      const predictionLabels = prediction.as1D().argMax().dataSync();
      const predictionLabel = predictionLabels[0];
      console.log("Expected Label", expectedLabel);
      console.log("Predicted Label", pokemon[predictionLabel]);

      prediction.dispose();
      activatedImage.dispose();
    });

(async () => {
  console.clear();
  console.log(HEADER.join("\n"));
  const modelPath = path.join(MODEL_DIR, "model.json");
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  const pretrainedModel = await buildPretrainedModel();

  await pEachSeries(
    ["bulbasaur", "squirtle", "charmander", "mankey"],
    (pokemon) => {
      const pkmnPath = path.join(process.cwd(), "test", `${pokemon}.png`);
      makePrediction(pretrainedModel, model, pkmnPath, pokemon);
    }
  );
})();
