import { t } from "./i18n.js"

// IDs can be found under game.system.id
export const successLines = new Map([
  ["alienrpg", "TERMINAL.Flavor.Alien"],
  ["starwarsffg", "TERMINAL.Flavor.StarWars"],
  ["blade-runner", "TERMINAL.Flavor.BladeRunner"],
  ["cyberpunk-red-core", "TERMINAL.Flavor.Cyberpunk"],
  ["fallout", "TERMINAL.Flavor.Fallout"],
  ["sfrpg", "TERMINAL.Flavor.Starfinder"],
  ["lancer", "TERMINAL.Flavor.Lancer"],
  // ["impmal", "Warhammer 40,000: Imperium Maledictum"],
  // ["wrath-and-glory", "Warhammer 40,000: Wrath & Glory"],
])

export const stylePresets = new Map([
  ["alienrpg", "alien"],
  ["starwarsffg", "star-wars"],
  ["blade-runner", "blade-runner"],
  ["cyberpunk-red-core", "cyberpunk"],
  ["fallout", "fallout"],
  ["impmal", "warhammer-man"],
  ["wrath-and-glory", "warhammer-man"],
  ["lancer", "lancer-union"],
])

export const ASCII = {
  // slant
  ALIEN: `
   _____                      __        ___       __
  / ___/___ _   ______ ______/ /_____  / (_)___  / /__
  \\__ \\/ _ \\ | / / __ \`/ ___/ __/ __ \\/ / / __ \\/ //_/
 ___/ /  __/ |/ / /_/ (__  ) /_/ /_/ / / / / / / ,<
/____/\\___/|___/\\__,_/____/\\__/\\____/_/_/_/ /_/_/|_|
`,
  ATLAS: `<h1 style="font-family: inherit">Map Downloaded</h1><pre>
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣄⠀⠀⠀⠀⣠⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣤⡀⠀⢀⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀⣀⣤⣶⡟⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠈⣻⣾⣿⣿⣿⡿⠟⠛⠛⠛⠛⠻⢿⣿⣿⣿⡿⣻⡟⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣴⣿⣿⣿⠟⠁⠀⠀⠀⠀⢀⣠⣴⣿⣿⡿⠋⣼⣿⣦⠀⠀⠀⠀⠀
⠀⢠⣄⣀⣼⣿⣿⡿⠁⠀⠀⠀⣀⣤⣾⣿⣿⣿⡿⠋⢀⣼⢿⣿⣿⣧⣀⣠⡄⠀
⠀⠀⠀⠙⣿⣿⣿⠁⠀⠀⠀⣼⠛⢿⣿⣿⡿⠋⠀⢀⡾⠃⠈⣿⣿⣿⠋⠀⠀⠀
⠀⠀⠀⠀⣿⣿⣿⠀⠀⢀⣾⠃⠀⠀⢙⡋⠀⠀⢠⡿⠁⠀⠀⣿⣿⣿⠀⠀⠀⠀
⠀⠀⠀⣠⣿⣿⣿⡀⢀⡾⠁⠀⢀⣴⣿⣿⣦⣠⡟⠁⠀⠀⢀⣿⣿⣿⣄⠀⠀⠀
⠀⠘⠋⠉⢻⣿⣿⣷⡿⠁⢀⣴⣿⣿⣿⡿⠟⠋⠀⠀⠀⢀⣾⣿⣿⡟⠉⠙⠃⠀
⠀⠀⠀⠀⠀⢻⣿⡟⢀⣴⣿⣿⠿⠋⠁⠀⠀⠀⠀⢀⣴⣿⣿⣿⡟⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣼⢟⣴⣿⣿⣿⣷⣦⣤⣤⣤⣤⣴⣶⣿⣿⣿⡿⣯⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣼⠿⠛⠉⠉⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠉⠀⠈⠛⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠋⠀⠉⠉⠀⠙⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
</pre>`,
  // doom
  EYE: `<h1 style="font-family: inherit">Control Granted</h1><pre>
  ⠀⠀⠀⠀⣀⣤⡤⠤⠶⠤⢤⣄⣀⠀⠀⠀
  ⠀⢀⡴⠛⣉⣠⣤⣴⣶⠶⣤⣄⡉⠳⢄⠀
  ⠀⢀⣴⠞⢹⠃⢰⣿⣧⣀⡄⠙⡟⢷⡀⠀
  ⢠⣾⠁⠀⢸⡄⠘⠿⣿⠿⠃⢰⠇⠀⢻⣄
  ⠹⠿⣤⡀⠀⠙⠶⢤⣤⡤⠖⠃⢀⣠⡾⠟
  ⠀⠀⠀⢉⠛⠒⠶⠶⠶⠶⠶⠚⢋⠁⠀⠀
</pre>`,
  MOTION: `<h1 style="font-family: inherit">Detecting Motion</h1><pre>
  ⠀⠀⠀⠀⣀⣤⡤⠤⠶⠤⢤⣄⣀⠀⠀⠀
  ⠀⢀⡴⠛⣉⣠⣤⣴⣶⠶⣤⣄⡉⠳⢄⠀
  ⠀⢀⣴⠞⢹⠃⢰⣿⣧⣀⡄⠙⡟⢷⡀⠀
  ⢠⣾⠁⠀⢸⡄⠘⠿⣿⠿⠃⢰⠇⠀⢻⣄
  ⠹⠿⣤⡀⠀⠙⠶⢤⣤⡤⠖⠃⢀⣠⡾⠟
  ⠀⠀⠀⢉⠛⠒⠶⠶⠶⠶⠶⠚⢋⠁⠀
</pre>`,
  DOOR_OPEN: `<h1 style="font-family: inherit">Door Opened</h1><pre>
 ______________
|\\ ___________ /|
| |  /|,| |   | |
| | |,x,| |   | |
| | |,x,' |   | |
| | |,x   ,   | |
| | |/    |   | |
| |    /] ,   | |
| |   [/ ()   | |
| |       |   | |
| |       |   | |
| |       |   | |
| |      ,'   | |
| |   ,'      | |
|_|,'_________|_|
</pre>`,
  DOOR_LOCK: `<h1 style="font-family: inherit">Door Locked</h1><pre>
 ______________
|\\ ___________ /|
| |  _ _ _ _  | |
| | | | | | | | |
| | |-+-+-+-| | |
| | |-+-+=+%| | |
| | |_|_|_|_| | |
| |    ___    | |
| |   [___] ()| |
| |         ||| |
| |         ()| |
| |           | |
| |           | |
| |           | |
|_|___________|_|
</pre>`,
  POWER_ON: `<h1 style="font-family: inherit">Power On</h1><pre>
         |
 \\     _____     /
     /       \\
    (         )
-   ( ))))))) )   -
     \\ \\   / /
      \\|___|/
  /    |___|    \\
       |___|
       |___|
</pre>`,
  POWER_OFF: `<h1 style="font-family: inherit">Power Off</h1><pre style="margin-top: 35px">

       _____
     /       \\
    (         )
    ( ))))))) )
     \\ \\   / /
      \\|___|/
       |___|
       |___|
       |___|
</pre>`,
  SSH: `<h1 style="font-family: inherit">Connection Established</h1><pre>
+-----[SECURE]----+
|       .++       |
|       .+..     .|
|     . .   . . ..|
|    . .     .E.. |
|     ...S     .  |
|      o+.        |
|     +..o        |
|  o B .o.        |
| . + +..         |
+-----[SHELL]-----+
</pre>`,
  BATTERY: `<h1 style="font-family: inherit">Equipment Charged</h1><pre>
████████████████████████████████████
██                                ██
██  ████  ████  ████  ████  ████  ████
██  ████  ████  ████  ████  ████  ████
██  ████  ████  ████  ████  ████  ████
██  ████  ████  ████  ████  ████  ████
██  ████  ████  ████  ████  ████  ████
██                                ██
████████████████████████████████████
</pre>`,
  // Alligator
  MACRO: `<pre style="font-size: 0.7em">
..####....####...#####...######..#####...######.
.##......##..##..##..##....##....##..##....##...
..####...##......#####.....##....#####.....##...
.....##..##..##..##..##....##....##........##...
..####....####...##..##..######..##........##...
................................................
.######..##..##..######...####...##..##..######..######..#####..
.##.......####...##......##..##..##..##....##....##......##..##.
.####......##....####....##......##..##....##....####....##..##.
.##.......####...##......##..##..##..##....##....##......##..##.
.######..##..##..######...####....####.....##....######..#####..
................................................................
</pre>`,
  MACRO_DENY: `<pre style="font-size: 0.8em">
..%%%%...%%%%..%%%%%..%%%%%%.%%%%%..%%%%%%.
.%%.....%%..%%.%%..%%...%%...%%..%%...%%...
..%%%%..%%.....%%%%%....%%...%%%%%....%%...
.....%%.%%..%%.%%..%%...%%...%%.......%%...
..%%%%...%%%%..%%..%%.%%%%%%.%%.......%%...
...........................................
.%%..%%.%%..%%.%%%%%%.%%..%%.%%%%%%..%%%%.........
.%%..%%.%%%.%%.%%......%%%%..%%.....%%..%%........
.%%..%%.%%.%%%.%%%%.....%%...%%%%...%%.....%%%%%%.
.%%..%%.%%..%%.%%......%%%%..%%.....%%..%%........
..%%%%..%%..%%.%%%%%%.%%..%%.%%%%%%..%%%%.........
..................................................
.%%..%%.%%%%%%..%%%%..%%%%%..%%.....%%%%%%.
.%%..%%...%%...%%..%%.%%..%%.%%.....%%.....
.%%..%%...%%...%%%%%%.%%%%%..%%.....%%%%...
.%%..%%...%%...%%..%%.%%..%%.%%.....%%.....
..%%%%....%%...%%..%%.%%%%%..%%%%%%.%%%%%%.
...........................................
</pre>`,
  // big money-se
  FALLOUT: `
  _______           __        ______
 |       \\         |  \\      /      \\
 | $$$$$$$\\ ______ | $$____ |  $$$$$$\\ ______
 | $$__| $$/      \\| $$    \\| $$   \\$$/      \\
 | $$    $|  $$$$$$| $$$$$$$| $$     |  $$$$$$\\
 | $$$$$$$| $$  | $| $$  | $| $$   __| $$  | $$
 | $$  | $| $$__/ $| $$__/ $| $$__/  | $$__/ $$
 | $$  | $$\\$$    $| $$    $$\\$$    $$\\$$    $$
  \\$$   \\$$ \\$$$$$$ \\$$$$$$$  \\$$$$$$  \\$$$$$$
`,
  CYBERPUNK: `
   ___   __            _
  / _ | / /_____ _____(_)
 / __ |/  '_/ _ \`/ __/ /
/_/ |_/_/\\_\\\\_,_/_/ /_/
`,
  // aligator
  STARWARS: `
      ::::::::   ::::::::  :::::::::  :::::::::: ::::    ::: :::::::::: :::::::::::
    :+:    :+: :+:    :+: :+:    :+: :+:        :+:+:   :+: :+:            :+:
   +:+        +:+    +:+ +:+    +:+ +:+        :+:+:+  +:+ +:+            +:+
  +#+        +#+    +:+ +#++:++#:  +#++:++#   +#+ +:+ +#+ +#++:++#       +#+
 +#+        +#+    +#+ +#+    +#+ +#+        +#+  +#+#+# +#+            +#+
#+#    #+# #+#    #+# #+#    #+# #+#        #+#   #+#+# #+#            #+#
########   ########  ###    ### ########## ###    #### ##########     ###
`,
  // small slant
  ARASAKA: `
    ___                     __
   / _ | _______ ____ ___ _/ /_____ _
  / __ |/ __/ _ \`(_-</ _ \`/  '_/ _ \`/
 /_/ |_/_/  \\_,_/___/\\_,_/_/\\_\\\\_,_/
`,
  // ticks
  WARHAMMER_MACHINE: `
  ___/\\/\\/\\/\\________________________________/\\/\\___
  _/\\/\\____/\\/\\__/\\/\\/\\__/\\/\\____/\\/\\/\\/\\___________
  _/\\/\\____/\\/\\__/\\/\\/\\/\\/\\/\\/\\__/\\/\\__/\\/\\__/\\/\\___
  _/\\/\\____/\\/\\__/\\/\\__/\\__/\\/\\__/\\/\\__/\\/\\__/\\/\\___
  ___/\\/\\/\\/\\____/\\/\\______/\\/\\__/\\/\\__/\\/\\__/\\/\\/\\_
  __________________________________________________
`,

  // slant relief
  BLADERUNNER: `
__/\\\\\\______________/\\\\\\_________________/\\\\\\\\\\\\_____/\\\\\\\\\\\\________________________________________________
 _\\/\\\\\\_____________\\/\\\\\\________________\\////\\\\\\____\\////\\\\\\________________________________________________
  _\\/\\\\\\_____________\\/\\\\\\___________________\\/\\\\\\_______\\/\\\\\\________________________________________________
   _\\//\\\\\\____/\\\\\\____/\\\\\\___/\\\\\\\\\\\\\\\\\\_______\\/\\\\\\_______\\/\\\\\\_____/\\\\\\\\\\\\\\\\\\________/\\\\\\\\\\\\\\\\_____/\\\\\\\\\\\\\\\\__
    __\\//\\\\\\__/\\\\\\\\\\__/\\\\\\___\\////////\\\\\\______\\/\\\\\\_______\\/\\\\\\____\\////////\\\\\\_____/\\\\\\//////____/\\\\\\/////\\\\\\_
     ___\\//\\\\\\/\\\\\\/\\\\\\/\\\\\\______/\\\\\\\\\\\\\\\\\\\\_____\\/\\\\\\_______\\/\\\\\\______/\\\\\\\\\\\\\\\\\\\\___/\\\\\\__________/\\\\\\\\\\\\\\\\\\\\\\__
      ____\\//\\\\\\\\\\\\//\\\\\\\\\\______/\\\\\\/////\\\\\\_____\\/\\\\\\_______\\/\\\\\\_____/\\\\\\/////\\\\\\__\\//\\\\\\________\\//\\\\///////___
       _____\\//\\\\\\__\\//\\\\\\______\\//\\\\\\\\\\\\\\\\/\\\\__/\\\\\\\\\\\\\\\\\\__/\\\\\\\\\\\\\\\\\\_\\//\\\\\\\\\\\\\\\\/\\\\__\\///\\\\\\\\\\\\\\\\__\\//\\\\\\\\\\\\\\\\\\\\_
        ______\\///____\\///________\\////////\\//__\\/////////__\\/////////___\\////////\\//_____\\////////____\\//////////__
`,
}

export function configurePresetLocalization() {
  const headings = {
    ATLAS: "TERMINAL.Action.MapDownloaded",
    EYE: "TERMINAL.Action.ControlGranted",
    MOTION: "TERMINAL.Action.DetectingMotion",
    DOOR_OPEN: "TERMINAL.Action.DoorOpened",
    DOOR_LOCK: "TERMINAL.Action.DoorLocked",
    POWER_ON: "TERMINAL.Action.PowerOn",
    POWER_OFF: "TERMINAL.Action.PowerOff",
    SSH: "TERMINAL.Action.ConnectionEstablished",
    BATTERY: "TERMINAL.Action.EquipmentCharged",
  }
  for (const [name, key] of Object.entries(headings)) {
    const previous = ASCII[name]
    ASCII[name] = previous.replace(/(<h1[^>]*>).*?(<\/h1>)/, `$1${t(key)}$2`)
    for (const style of Object.values(defaultStyles)) {
      if (style.ascii === previous) style.ascii = ASCII[name]
    }
  }
}

export const defaultStyles = {
  fallout: {
    ascii: ASCII.FALLOUT,
    shadow: "#19572e",
    base: "#23a84f",
    highlight: "#b6fab6",
    name: "Fallout",
    click: "/modules/terminal/audio/fallout_click.mp3",
    close: "/modules/terminal/audio/fallout_close.mp3",
    startup: "/modules/terminal/audio/fallout_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_2.webp",
    borderSlice: 35,
    uuid: "fallout",
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  alien: {
    ascii: ASCII.ALIEN,
    shadow: "#19572e",
    base: "#23a84f",
    highlight: "#b6fab6",
    name: "Alien",
    click: "modules/terminal/audio/alien_click.mp3",
    close: "modules/terminal/audio/alien_close.mp3",
    startup: "modules/terminal/audio/alien_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    borderSlice: 35,
    uuid: "alien",
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "alien-wayland": {
    ascii: ASCII.ALIEN,
    shadow: "#443c03",
    base: "#e7c80a",
    highlight: "#ece3a7",
    name: "Alien Wayland",
    click: "modules/terminal/audio/alien_click.mp3",
    close: "modules/terminal/audio/alien_close.mp3",
    startup: "modules/terminal/audio/alien_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    background: "modules/terminal/background/yellow_code.webm",
    splashFile: "modules/terminal/background/wayland.webm",
    borderSlice: 35,
    opacity: 0.55,
    uuid: "alien-wayland",
    readOnly: true,
    effectScan: false,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "star-wars": {
    ascii: ASCII.STARWARS,
    shadow: "#000d1c", // 0, 13, 28
    base: "#409bbf", // 64, 155, 191
    highlight: "#8bcbf7",
    name: "Star Wars",
    background: "modules/terminal/background/blue.mp4",
    click: "modules/terminal/audio/star_wars_click.mp3",
    close: "modules/terminal/audio/star_wars_close.mp3",
    startup: "modules/terminal/audio/star_wars_startup.mp3",
    uuid: "star-wars",
    borderImage: "modules/terminal/background/circuit_1.webp",
    borderSlice: 35,
    opacity: 0.4,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "lancer-union": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#450303",
    base: "#e60a0a",
    highlight: "#eca7a7",
    name: "Lancer Union",
    background: "modules/terminal/background/red_rainbow.webm",
    click: "modules/terminal/audio/star_wars_click.mp3",
    close: "modules/terminal/audio/star_wars_close.mp3",
    startup: "modules/terminal/audio/star_wars_startup.mp3",
    uuid: "lancer-union",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/lancer_union.webm",
    borderSlice: 35,
    opacity: 0.7,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "lancer-harrison-armory": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#340934",
    base: "#771677",
    highlight: "#ffb3ff",
    name: "Lancer Harrison Armory",
    background: "modules/terminal/background/digits.webm",
    click: "modules/terminal/audio/star_wars_click.mp3",
    close: "modules/terminal/audio/star_wars_close.mp3",
    startup: "modules/terminal/audio/star_wars_startup.mp3",
    uuid: "lancer-harrison-armory",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/harrison_armory.webm",
    borderSlice: 35,
    opacity: 0.1,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "lancer-ipsn": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#08273f",
    base: "#2691e3",
    highlight: "#c6e4fb",
    name: "Lancer IPS-Northstar",
    background: "modules/terminal/background/bubble_matrix.webm",
    click: "modules/terminal/audio/star_wars_click.mp3",
    close: "modules/terminal/audio/star_wars_close.mp3",
    startup: "modules/terminal/audio/star_wars_startup.mp3",
    uuid: "lancer-ipsn",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/ipsn.webm",
    borderSlice: 35,
    opacity: 0.4,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "lancer-horus": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#0a1e0f",
    base: "#338e49",
    highlight: "#dcfee5",
    name: "Lancer Horus",
    background: "modules/terminal/background/error_background.gif",
    click: "modules/terminal/audio/star_wars_click.mp3",
    close: "modules/terminal/audio/star_wars_close.mp3",
    startup: "modules/terminal/audio/star_wars_startup.mp3",
    uuid: "lancer-horus",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/horus.webm",
    borderSlice: 35,
    opacity: 0.9,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "lancer-ssc": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#181d25",
    base: "#ffeaa3",
    highlight: "#feffd6",
    name: "Lancer Smith-Shimano",
    background: "modules/terminal/background/blue_eyes.webm",
    click: "modules/terminal/audio/star_wars_click.mp3",
    close: "modules/terminal/audio/star_wars_close.mp3",
    startup: "modules/terminal/audio/star_wars_startup.mp3",
    uuid: "lancer-ssc",
    borderImage: "modules/terminal/background/metal_door_2.webp",
    splashFile: "modules/terminal/background/lancer_ssc.webm",
    borderSlice: 20,
    opacity: 0.8,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  cyberpunk: {
    ascii: ASCII.CYBERPUNK,
    background: "modules/terminal/background/cyberpunk.webp",
    splashFile: "modules/terminal/background/cyberpunk.webp",
    shadow: "#5a0000",
    base: "#ff0000",
    highlight: "#ff9595",
    name: "Cyberpunk",
    click: "modules/terminal/audio/cyberpunk_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/cyberpunk_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    borderSlice: 35,
    uuid: "cyberpunk",
    opacity: 0.8,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "cyberpunk-arasaka": {
    ascii: ASCII.ARASAKA,
    background: "modules/terminal/background/files_background.webm",
    shadow: "#195756",
    base: "#2eb8c2",
    highlight: "#ddedfd",
    name: "Cyberpunk Arasaka",
    click: "modules/terminal/audio/cyberpunk_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/cyberpunk_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/cyberpunk.webm",
    borderSlice: 35,
    uuid: "cyberpunk-arasaka",
    opacity: 1,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "warhammer-machine": {
    ascii: ASCII.WARHAMMER_MACHINE,
    background: "modules/terminal/background/omnissiah.webp",
    shadow: "#450303",
    base: "#e60a0a",
    highlight: "#eca7a7",
    name: "Warhammer Adeptus Mechanicus",
    click: "modules/terminal/audio/cyberpunk_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/cyberpunk_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    splashFile: "modules/terminal/background/warhammer_machine.webm",
    borderSlice: 35,
    uuid: "warhammer-machine",
    opacity: 0.5,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "warhammer-man": {
    ascii: ASCII.WARHAMMER_MACHINE,
    background: "modules/terminal/background/gold_1.webp",
    shadow: "#453603",
    base: "#e6b60a",
    highlight: "#ecd9a7",
    name: "Warhammer Imperium of Man",
    click: "modules/terminal/audio/cyberpunk_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/cyberpunk_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_2.webp",
    splashFile: "modules/terminal/background/warhammer_man.webm",
    borderSlice: 35,
    uuid: "warhammer-man",
    opacity: 0.3,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "blade-runner": {
    ascii: ASCII.BLADERUNNER,
    shadow: "#3d3d3d",
    base: "#ffff92",
    highlight: "#ffffff",
    name: "Blade Runner",
    background: "modules/terminal/background/blade_runner.webp",
    splashFile: "modules/terminal/background/blade_runner.webp",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/blade_runner_close.mp3",
    startup: "modules/terminal/audio/blade_runner_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    borderSlice: 35,
    uuid: "blade-runner",
    opacity: 0.6,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "generic-red": {
    ascii: ASCII.STARWARS,
    shadow: "#340000",
    base: "#b52a2a",
    highlight: "#f98686",
    name: "Generic Red",
    background: "modules/terminal/background/red.mp4",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    splashFile: "modules/terminal/background/beat_roots.webm",
    borderImage: "modules/terminal/background/scifi_red.webp",
    borderSlice: 35,
    uuid: "generic-red",
    opacity: .8,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "generic-green": {
    ascii: ASCII.STARWARS,
    shadow: "#19572e",
    base: "#23a84f",
    highlight: "#b6fab6",
    name: "Generic Green",
    background: "modules/terminal/background/green.mp4",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    borderSlice: 35,
    uuid: "generic-green",
    opacity: 0.6,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "generic-mint": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#062815",
    base: "#12e293",
    highlight: "#e3ede7",
    name: "Generic Mint",
    background: "modules/terminal/background/files_background.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    splashFile: "modules/terminal/background/helicopter_cube.gif",
    borderSlice: 20,
    uuid: "generic-mint",
    opacity: 0.9,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "generic-red-2": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#280606",
    base: "#e21212",
    highlight: "#ffb8b8",
    name: "Generic Red 2",
    background: "modules/terminal/background/sandsifter.gif",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    splashFile: "modules/terminal/background/file_web.webm",
    borderSlice: 20,
    uuid: "generic-red-2",
    opacity: 0.4,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "generic-purple": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#1a0628",
    base: "#781fc1",
    highlight: "#ecb8ff",
    name: "Generic Purple",
    background: "modules/terminal/background/fiber_pulse.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    splashFile: "modules/terminal/background/white_ice.webm",
    borderSlice: 20,
    uuid: "generic-purple",
    opacity: .3,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "generic-blue": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#05191f",
    base: "#20a2d9",
    highlight: "#c2d4ff",
    name: "Generic Blue",
    background: "modules/terminal/background/skull_1.gif",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    splashFile: "modules/terminal/background/dark_city.mp4",
    borderSlice: 20,
    uuid: "generic-blue",
    opacity: 1,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "generic-red-3": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#1f0511",
    base: "#d92033",
    highlight: "#ff9e9e",
    name: "Generic Red 3",
    background: "modules/terminal/background/numeral_matrix.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    splashFile: "modules/terminal/background/sys_reboot.webm",
    borderSlice: 20,
    uuid: "generic-red-3",
    opacity: .2,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "lain": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#4f1212",
    base: "#e42525",
    highlight: "#febebe",
    name: "Lain",
    background: "modules/terminal/background/lain.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_square_1.webp",
    splashFile: "modules/terminal/background/hacked_splash.gif",
    borderSlice: 20,
    uuid: "lain",
    opacity: .5,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "generic-pink": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#6e0c45",
    base: "#ff007b",
    highlight: "#ffe0f7",
    name: "Generic Pink",
    background: "modules/terminal/background/black_ice.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/spin_unlock_splash.gif",
    borderSlice: 20,
    uuid: "generic-pink",
    opacity: .2,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "generic-cyan": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#0c6e6d",
    base: "#20a5b6",
    highlight: "#e0feff",
    name: "Generic Cyan",
    background: "modules/terminal/background/numeral_matrix.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/cube_splash.webm",
    borderSlice: 20,
    uuid: "generic-cyan",
    opacity: .2,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "generic-orange": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#3c280b",
    base: "#ff8800",
    highlight: "#fff9eb",
    name: "Generic Orange",
    background: "modules/terminal/background/mantis_background.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/crosshair_splash.gif",
    borderSlice: 20,
    uuid: "generic-orange",
    opacity: .9,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "generic-yellow": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#3c340b",
    base: "#ffd500",
    highlight: "#fffcbd",
    name: "Generic Yellow",
    background: "modules/terminal/background/skull_1.gif",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/crosshair_splash.gif",
    borderSlice: 20,
    uuid: "generic-yellow",
    opacity: .9,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
  "generic-red-4": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#3c0b0b",
    base: "#ff0000",
    highlight: "#ffbdbd",
    name: "Generic Red 4",
    background: "modules/terminal/background/skull_2.gif",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/cyber_hack_splash.webm",
    borderSlice: 20,
    uuid: "generic-red-4",
    opacity: 1,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: false,
  },
  "generic-turquoise": {
    ascii: ASCII.WARHAMMER_MACHINE,
    shadow: "#0a2e2b",
    base: "#30D5C8",
    highlight: "#e0fffc",
    name: "Generic Turquoise",
    background: "modules/terminal/background/skull_3.webm",
    click: "modules/terminal/audio/blade_runner_click.mp3",
    close: "modules/terminal/audio/cyberpunk_close.mp3",
    startup: "modules/terminal/audio/generic_startup.mp3",
    borderImage: "modules/terminal/background/metal_door_1.webp",
    splashFile: "modules/terminal/background/timeline_splash.webm",
    borderSlice: 20,
    uuid: "generic-turquoise",
    opacity: .8,
    readOnly: true,
    effectScan: true,
    effectScramble: true,
    effectGlitch: true,
    showASCIILoading: true,
  },
}
