const pokemon = require("./pokemon.json");
const tf = require("@tensorflow/tfjs");
const tfnode = require("@tensorflow/tfjs-node");
const fs = require("fs").promises;
const path = require("path");
const pReduce = require("p-reduce");
const _ = require("lodash");
const { DATA_DIR, DATA_LIMIT, HEADER } = require("./constants");

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
        }));
    })
  ).then((data) => data.flat());

const resizeImage = (image) => tf.image.resizeBilinear(image, [224, 224]);

const batchImage = (image) => {
  // Expand our tensor to have an additional dimension, whose size is 1
  const batchedImage = image.expandDims(0);

  // Turn pixel data into a float between -1 and 1.
  return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
};

const cropImage = (img) => {
  const width = img.shape[0];
  const height = img.shape[1];

  // use the shorter side as the size to which we will crop
  const shorterSide = Math.min(img.shape[0], img.shape[1]);

  // calculate beginning and ending crop points
  const startingHeight = (height - shorterSide) / 2;
  const startingWidth = (width - shorterSide) / 2;
  const endingHeight = startingHeight + shorterSide;
  const endingWidth = startingWidth + shorterSide;

  // return image data cropped to those points
  return img.slice(
    [startingWidth, startingHeight, 0],
    [endingWidth, endingHeight, 3]
  );
};

const loadImage = async (src) => {
  try {
    const buffer = await fs.readFile(src);
    return tfnode.node.decodeImage(buffer);
  } catch (error) {
    throw Error(src, error);
  }
};

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
  pReduce(imagePaths, async (data, path, index) => {
    console.clear();
    console.log(HEADER.join("\n"));
    console.log("Current Pokémon:", toTitleCase(labels[index]));
    console.log("Load image", index + 1, "of", imagePaths.length);
    const loadedImage = await loadImage(path);

    return tf.tidy(() => {
      const croppedImage = cropImage(loadedImage);
      const resizedImage = resizeImage(croppedImage);
      const batchedImage = batchImage(resizedImage);
      const prediction = pretrainedModel.predict(batchedImage);

      if (data) {
        const newData = data.concat(prediction);
        data.dispose();
        return newData;
      }

      return tf.keep(prediction);
    });
  });

const oneHot = (labelIndex, classLength) => {
  return tf.tidy(() =>
    tf.oneHot(tf.tensor1d([labelIndex]).toInt(), classLength)
  );
};

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

const addLabels = (labels) => {
  return tf.tidy(() => {
    const classes = getLabelsAsObject(labels); // _.uniq(labels);
    const numberOfClasses = Object.keys(classes).length;

    let ys;
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i];
      const labelIndex = classes[label];
      const y = oneHot(labelIndex, numberOfClasses);
      if (i === 0) {
        ys = y;
      } else {
        ys = ys.concat(y, 0);
      }
    }
    return { ys, numberOfClasses };
  });
};

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

const makePrediction = (pretrainedModel, model, image, expectedLabel) =>
  loadImage(image)
    .then((loadedImage) => {
      return loadAndProcessImage(loadedImage);
    })
    .then((loadedImage) => {
      const activatedImage = pretrainedModel.predict(loadedImage);
      loadedImage.dispose();
      return activatedImage;
    })
    .then((activatedImage) => {
      const prediction = model.predict(activatedImage);
      const predictionLabel = prediction.as1D().argMax().dataSync()[0];

      console.log("Expected Label", expectedLabel);
      console.log("Predicted Label", predictionLabel);

      prediction.dispose();
      activatedImage.dispose();
    });

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

  // make predictions
  // makePrediction(pretrainedModel, model, blue3, "0");
  // makePrediction(pretrainedModel, model, red3, "1");
})();
