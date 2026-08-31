const MODULE_ID = 'sf2e_compendium'; // Change this ID!

// No need to change the code below this line, but it’s your module so do it if you want!
Hooks.once('babele.init', (babele) => {
  babele.register({
    module: MODULE_ID,
    lang: 'cn',
    dir: 'compendium/cn',
  });
});
