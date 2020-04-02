const pEachSeries = require("p-each-series");
const fs = require("fs");
const playwright = require("playwright");
const pokemon = require("./pokemon.json");

const getImageSources = async (page, pokemonName) => {
  try {
    await page.goto(
      `https://duckduckgo.com/?q=${pokemonName}&t=h_&ia=images&iax=images`
    );

    const images = await page.$$(".tile");
    console.debug("Saving images for", pokemonName);
    await Promise.all(
      images.map(async (img, index) => {
        const type = "jpeg";
        const fileName = `${index}.${type}`;
        const basePath = `${__dirname}/results/${pokemonName}`;
        const path = `${basePath}/${fileName}`;

        if (!fs.existsSync(basePath)) {
          fs.mkdirSync(basePath);
        }

        await img.screenshot({
          path,
          type
        });
        console.debug(
          "Done saving image",
          pokemonName,
          index,
          "\n",
          "-------------------------"
        );
      })
    );
  } catch (error) {
    Promise.reject(error);
  }
};

(async () => {
  const browser = await playwright.chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  await pEachSeries(pokemon, pkmn => getImageSources(page, pkmn)).catch(e => {
    throw Error(e);
  });
  await browser.close();
})();
