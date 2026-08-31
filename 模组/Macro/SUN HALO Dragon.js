let token = await Tagger.getByTag("gob 1")[0].object;
let target1 = await Tagger.getByTag("gob 2")[0].object;
let target2 = await Tagger.getByTag("gob 3")[0].object;
let target3 = await Tagger.getByTag("gob 4")[0].object;

let tile = await Tagger.getByTag("drag")[0];
let pos1 = await Tagger.getByTag("pos1")[0];
let pos2 = await Tagger.getByTag("pos2")[0];

//FLASH IMPACT FRAME HERE
let impact = false;
//SPEEDLINE TOGGLE
let screen = true;

new Sequence()

  .animation()
  .on(token)
  .teleportTo({x:7424, y:6144})
  .opacity(0)
  
  .effect()
  .copySprite(token)
  .atLocation(token)
  .animateProperty("sprite", "position.x", { from: -0, to: -9, duration: 500, gridUnits: true, ease: "easeOutQuint",delay: 2000+150+500 })
  .duration(10000)

  .effect()
  .copySprite(token)
  .atLocation(token)
  .animateProperty("sprite", "position.x", { from: -0, to: -9, duration: 500, gridUnits: true, ease: "easeOutQuint",delay: 2000+150+500 })
  .duration(2000+150+500)
  .fadeIn(100,{delay: 2500+150})
  .fadeOut(250)
      .filter("Blur", { blurX: 15, blurY: 0 })

.wait(500)

  .effect()
  .file("eskie.screen_overlay.speed_lines.horizontal.02.redyellow")
  .screenSpace()
  .screenSpaceScale({fitX:true,fitY:true})
  .mirrorX()
  .fadeOut(500)
  .duration(2500)
  .playIf(screen)
  
  
     .effect()
        .file("jb2a.wind_stream.white")
        .name("Rage")
        .attachTo(token, {bindAlpha: false})
        .scaleToObject()
        .rotate(90)
        .opacity(1)
        .filter("ColorMatrix", {saturate: 1})
        .tint("#FF5733")
        .private()
 //.mask()
  .duration(2000)
  .fadeOut(250)
  .zIndex(5)

    .effect()
  .file("eskie.aura.token.generic.01.redorange")
    .atLocation(token)
  .scaleToObject(2.1)
    .zIndex(0.1)
  .belowTokens()
    .animateProperty("sprite", "position.x", { from: -0, to: -9, duration: 500, gridUnits: true, ease: "easeOutQuint",delay: 2000+50 })
    .animateProperty("sprite", "rotation", { from: -0, to: 90, duration: 50, ease: "easeOutQuint",delay: 2000+50 })

  
  .effect()
    .file("eskie.fire.03.redorange")
    .atLocation(token, {offset:{x:-0.3, y:-0.15}, gridUnits:true})
    .scaleToObject(0.5)
  .playbackRate(1.2)
  .mirrorX()
  .zIndex(1)
  
  .effect()
    .file("eskie.fire.03.redorange")
    .atLocation(token, {offset:{x:0.3, y:-0.2}, gridUnits:true})
    .scaleToObject(0.5)
  .playbackRate(1.2)
    .zIndex(1)
  
.macro("Sun Halo Targets", {delay:1000})

    .effect()
  .delay( 2100)
    .name(`Casting ${token.document.name}`)
    .file(canvas.scene.background.src)
    .filter("ColorMatrix", {saturate: 1, brightness: 0.6})
    .atLocation({x:(canvas.dimensions.width)/2,y:(canvas.dimensions.height)/2})
    .size({width:canvas.scene.width/canvas.grid.size, height:canvas.scene.height/canvas.grid.size}, {gridUnits: true})
  .duration(250)
    .filter("ColorMatrix", { brightness:0 })
    .belowTiles()
    .fadeOut(125)
    .fadeIn(125)
  .opacity(1)
    .spriteOffset({x:-canvas.scene.background.offsetX,y:-canvas.scene.background.offsetY})
  .playIf(impact)

  .effect()
    .delay( 2100)
  .file("eskie.environment.lighting.shine.01.rainbow")
  .atLocation(token)
  .scaleToObject(4)
.scaleIn(0, 250, {ease: "easeOutCubic"})
  .duration(250)
  .fadeOut(150)
  .playIf(impact)

  .effect()
  .delay(2100)
  .file("eskie.particle.04.orange")
    .atLocation(token)
  .scaleToObject(5)
  .animateProperty("sprite", "position.x", { from: -0, to: -7.5, duration: 500, gridUnits: true, ease: "easeOutQuint" })
  .belowTokens()
    .playIf(!impact)

  .effect()
    .delay(2000)
  .file("eskie.velocity.01.white")
.atLocation(token)
  .mirrorX()
  .scaleToObject(7.5)
  .opacity(0.5)
  .zIndex(10)
  .playbackRate(1.5)


    .canvasPan()
  .delay(2000)
    .shake({ duration: 250, strength: 1.5, rotation: false, fadeOut: 250 })



  
  .wait(2000)
  
.effect()
.file("eskie.fire.fire_dragon.01")
.atLocation(tile)
  .scale(0.75)
  .belowTokens()
  .playbackRate(1.25)


 
  .effect()
  .delay(150)
  .file("eskie.slice.01_ranged.color.rainbow")
  .atLocation(pos1)
  .stretchTo(pos2)
  .scale(1.5)
    .playbackRate(0.75)
  .zIndex(5)



  .wait(1250)

  .canvasPan()
    .shake({ duration: 500, strength: 1.5, rotation: false, fadeOut: 250 })

  .wait(10000)

  .animation()
  .on(token)
  .opacity(1)
  
.play()