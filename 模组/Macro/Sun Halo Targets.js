// Collect targets by tag
const tags = ["gob 2", "gob 3", "gob 4"];
const targets = tags
  .map(t => Tagger.getByTag(t)?.[0]?.object)
  .filter(Boolean);

const delay = 2000 + 250;

// Run the sequence for each target independently
for (const target of targets) {
  const gs = canvas.grid.size;
  const cx = target.center.x;
  const cy = target.center.y;

  new Sequence()

.wait(1000)
    
    .animation()
      .on(target)
      .opacity(0)

    .effect()
      .delay(delay-1000)
      .file("eskie.particle.03.orange")
      .atLocation(target,{randomOffset:0.5, gridUnits:true})
      .scaleToObject(2)
      .randomRotation()
      .zIndex(3)

        .effect()
      .delay(delay-1000)
.copySprite(target)
      .atLocation(target)
      .scaleToObject(1)
    .loopProperty("sprite", "position.x", { from: -0.05, to: 0.05, duration: 50, pingPong: true, gridUnits: true})
    .duration(250)
    .opacity(0.5)
      .zIndex(3)
    
    .effect()
      .delay(delay)
      .file("eskie.slice.01.white.rainbow")
      .atLocation(target)
      .scaleToObject(4)
      .rotate(-45)
      .zIndex(5)

    .effect()
      .delay(delay)
      .file("eskie.particle.03.orange")
      .atLocation(target)
      .scaleToObject(2)
      .randomRotation()
      .zIndex(4)

    .wait(500)

    // Top half mask copy
    .effect()
      .copySprite(target)
      .name(`${target.document.name}Top`)
      .scaleToObject()
      .atLocation(target)
      .shape("polygon", {
        lineSize: 1,
        lineColor: "#FF0000",
        fillColor: "#FF0000",
        points: [{ x: -1, y: -1 }, { x: 1, y: 1 }, { x: 1, y: -1 }],
        fillAlpha: 1,
        gridUnits: true,
        isMask: true,
        name: "test"
      })
      .moveTowards(
        { x: cx + gs * 0.25, y: cy - gs * 0.25 },
        { rotate: false, ease: "easeOutCubic", delay: delay }
      )
      .duration(3000)
      .persist()
      .fadeOut(1000)

    // Bottom half mask copy
    .effect()
      .copySprite(target)
      .name(`${target.document.name}Bottom`)
      .scaleToObject()
      .atLocation(target)
      .shape("polygon", {
        lineSize: 1,
        lineColor: "#FF0000",
        fillColor: "#FF0000",
        points: [{ x: -1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }],
        fillAlpha: 1,
        gridUnits: true,
        isMask: true,
        name: "test"
      })
      .duration(2500)
      .persist()
      .fadeOut(1000)

    // Burn mask top (moves with top slice)
    .effect()
      .delay(delay + 250)
      .file("eskie.burn.token_mask.orange.fast")
      .name(`${target.document.name}Top`)
      .scaleToObject(1.1)
      .atLocation({ x: cx + gs * 0.25, y: cy - gs * 0.25 })
      .shape("polygon", {
        lineSize: 1,
        lineColor: "#FF0000",
        fillColor: "#FF0000",
        points: [{ x: -1, y: -1 }, { x: 1, y: 1 }, { x: 1, y: -1 }],
        fillAlpha: 1,
        gridUnits: true,
        isMask: true,
        name: "test"
      })
      .moveTowards(
        { x: cx + gs * 0.25, y: cy - gs * 0.25 },
        { rotate: false, ease: "easeOutCubic", delay: 2000 }
      )

    // Burn mask bottom
    .effect()
      .delay(delay + 250)
      .file("eskie.burn.token_mask.orange.fast")
      .name(`${target.document.name}Bottom`)
      .scaleToObject(1.1)
      .atLocation(target)
      .shape("polygon", {
        lineSize: 1,
        lineColor: "#FF0000",
        fillColor: "#FF0000",
        points: [{ x: -1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }],
        fillAlpha: 1,
        gridUnits: true,
        isMask: true,
        name: "test"
      })
      .zIndex(1)

    // Embers top
    .effect()
      .delay(delay + 250)
      .file("eskie.burn.embers.orange")
      .name(`${target.document.name}Top`)
      .scaleToObject(1.5)
      .atLocation({ x: cx + gs * 0.25, y: cy - gs * 0.25 })
      .mirrorX()
      .fadeIn(500)
      .spriteOffset({ x: 0.3, y: -0.3 }, { gridUnits: true })

    // Embers bottom
    .effect()
      .delay(delay + 250)
      .file("eskie.burn.embers.orange")
      .name(`${target.document.name}Bottom`)
      .scaleToObject(1.5)
      .atLocation(target)
      .mirrorX()
      .fadeIn(500)
      .spriteOffset({ x: 0, y: 0 }, { gridUnits: true })
      .spriteRotation(-45)
      .zIndex(2)

    .wait(10000)

    .thenDo(() => {
      Sequencer.EffectManager.endEffects({ name: `${target.document.name}Top` });
      Sequencer.EffectManager.endEffects({ name: `${target.document.name}Bottom` });
    })

    .animation()
      .on(target)
      .opacity(1)

    .play();
}