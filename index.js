const pEachSeries = require("p-each-series");
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");
const pokemon = require("./pokemon.json");
const { DATA_DIR, HEADER } = require("./constants");

// Duck duck go
// const getUrl = query =>
//   `https://duckduckgo.com/?q=${quer}&t=h_&ia=images&iax=images`;
// const imageSelector = ".tile"

// Google
const getUrl = query => `https://www.google.com/search?q=${query}&tbm=isch`;
const imageSelector = "#islmp img";

const createDirIfMissing = dirName => {
  if (!fs.existsSync(dirName)) fs.mkdirSync(dirName);
};

const getAndSaveImages = async (page, pokemonName) => {
  try {
    await page.goto(getUrl(pokemonName));

    const type = "jpeg";
    const basePath = path.join(DATA_DIR, pokemonName);
    const images = await page.$$(imageSelector);

    createDirIfMissing(basePath);

    if (images.length > 0) {
      console.debug(
        "--------------------------------------------------",
        `\nSaving ${images.length} images for`,
        pokemonName,
      );
      await pEachSeries(images, async (img, index) => {
        const fileName = `${index}.${type}`;
        const path = `${basePath}/${fileName}`;

        await img.screenshot({
          path,
          type
        });
        console.debug("Saved", pokemonName, index);
      });
    } else {
      console.warn("No images found for", pokemonName);
      await page.screenshot({ path: `${basePath}/page_screen.png` });
    }
  } catch (error) {
    Promise.reject(error);
  }
};

(async () => {
  createDirIfMissing(DATA_DIR);
  const browser = await puppeteer.launch({
    defaultViewport: {
      width: 1440,
      height: 900
    }
  });
  const page = await browser.newPage();
  await pEachSeries(pokemon, pkmn => getAndSaveImages(page, pkmn)).catch(e => {
    throw Error(e);
  });
  await browser.close();
})();
