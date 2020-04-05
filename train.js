const pokemon = require("./pokemon.json");
const tf = require("@tensorflow/tfjs");
const tfnode = require("@tensorflow/tfjs-node");
const fs = require("fs").promises;
const path = require("path");
const pReduce = require("p-reduce");
const _ = require("lodash");
const { DATA_DIR, DATA_LIMIT, HEADER, FILE_LIMIT } = require("./constants");
const image = require("./lib/image");

const toTitleCase = (str) => str.charAt(0).toUpperCase() + str.slice(1);

const getData = () =>
  Promise.all(
    pokemon.map(async (pkmn) => {
      const dir = path.join(DATA_DIR, pkmn);
      const files = await fs.readdir(dir).catch((e) => {
        throw Error(e);
      });
      return files
        .filter((name) => name.includes(".jpeg"))
        .map((imageFile) => ({
          class: pkmn,
          path: `${dir}/${imageFile}`,
          label: pkmn,
        }))
        .slice(0, FILE_LIMIT);
    })
  ).then((data) => data.flat());

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

const loadImages = (imagePaths, labels, pretrainedModel) =>
  pReduce(
    imagePaths,
    async (data, path, index) => {
      console.clear();
      console.log(HEADER.join("\n"));
      console.log("Current Pokémon:", toTitleCase(labels[index]));
      console.log("Load image", index + 1, "of", imagePaths.length);
      const loadedImage = await image.loadSingle(path);

      return tf.tidy(() => {
        const croppedImage = image.crop(loadedImage);
        const resizedImage = image.resize(croppedImage);
        const batchedImage = image.batch(resizedImage);
        const prediction = pretrainedModel.predict(batchedImage);

        if (data) {
          const newData = data.concat(prediction);
          data.dispose();
          return newData;
        }

        return tf.keep(prediction);
      });
    },
    undefined
  );

const oneHot = (labelIndex, classLength) =>
  tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), classLength));

const getLabelsAsObject = (labels) => {
  let labelObject = {};
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    if (labelObject[label] === undefined) {
      // only assign it if we haven't seen it before
      labelObject[label] = Object.keys(labelObject).length;
    }
  }

  return labelObject;
};

const addLabels = (labels) =>
  tf.tidy(() => {
    const classes = getLabelsAsObject(labels);
    const numberOfClasses = Object.keys(classes).length;

    const ys = labels.reduce((acc, label, index) => {
      const labelIndex = classes[label];
      const y = oneHot(labelIndex, numberOfClasses);
      return index === 0 ? y : acc.concat(y, 0);
    }, undefined);
    return { ys, numberOfClasses };
  });

const getModel = (numberOfClasses) => {
  const model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: [7, 7, 256] }),
      tf.layers.dense({
        units: 100,
        activation: "relu",
        kernelInitializer: "varianceScaling",
        useBias: true,
      }),
      tf.layers.dense({
        units: numberOfClasses,
        kernelInitializer: "varianceScaling",
        useBias: false,
        activation: "softmax",
      }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};

(async () => {
  console.clear();
  console.log(HEADER.join("\n"));
  const data = await getData();
  const sliced = data.slice(0, DATA_LIMIT);
  const imagePaths = sliced.map(({ path }) => path);
  const labels = sliced.map(({ label }) => label);

  console.log("Load images");
  const pretrainedModel = await buildPretrainedModel();
  const xs = await loadImages(imagePaths, labels, pretrainedModel);

  console.log("Add labels");
  const { numberOfClasses, ys } = addLabels(labels);
  console.log(numberOfClasses);

  console.log("Get model");
  const model = getModel(numberOfClasses);

  console.log("Fit model");
  await model.fit(xs, ys, {
    epochs: 20,
    shuffle: true,
  });

  console.log("Save model");
  const outDir = path.join(process.cwd(), "dist");
  await model.save(`file:///${outDir}`);

  console.log("Training complete");
})();
