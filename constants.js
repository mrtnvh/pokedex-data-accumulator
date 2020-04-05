const path = require("path");

const DATA_DIR = path.join(process.cwd(), "data");
const DATA_LIMIT = Infinity;

// Add labels hoggs a lot of memory
const FILE_LIMIT = 30;
const HEADER = [
  "------------------------------------------",
  "Pokedex Model Trainer",
  "------------------------------------------",
];

module.exports = {
  DATA_DIR,
  DATA_LIMIT,
  HEADER,
  FILE_LIMIT,
};
