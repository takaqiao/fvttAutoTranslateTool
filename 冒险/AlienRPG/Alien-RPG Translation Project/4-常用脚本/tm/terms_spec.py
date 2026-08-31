# -*- coding: utf-8 -*-
"""英文闸正则清单。cs=True 表示大小写敏感（缩写与人名专用，避免 APC/Ash 误伤）。"""


def T(key, cat, en, cs=False, zh=None):
    return {"key": key, "cat": cat, "en": en, "cs": cs, "zh": zh}


TERMS = [
    # ---------------- 飞船 ----------------
    T('Nostromo', 'ship', r'\bNostromo\b'),
    T('Sulaco', 'ship', r'\bSulaco\b'),
    T('Covenant', 'ship', r'\bCovenant\b'),
    T('Prometheus', 'ship', r'\bPrometheus\b'),
    T('Torrens', 'ship', r'\bTorrens\b'),
    T('Anesidora', 'ship', r'\bAnesidora\b'),
    T('Corbelan', 'ship', r'\bCorbelan\b'),
    T('Cronus', 'ship', r'\bCronus\b'),
    T('Romulus', 'ship', r'\bRomulus\b'),
    T('Remus', 'ship', r'\bRemus\b'),
    T('Narcissus', 'ship', r'\bNarcissus\b'),
    T('Betty', 'ship', r'\bBetty\b'),

    # ---------------- 地点 ----------------
    T('LV-426', 'place', r'LV[\s\-]?426'),
    T('LV-223', 'place', r'LV[\s\-]?223'),
    T('Acheron', 'place', r'\bAcheron\b'),
    T("Hadley's Hope", 'place', r'\bHadley'),
    T('Fiorina 161', 'place', r'\bFiorina\b|\bFury\s?161\b'),
    T('Sevastopol', 'place', r'\bSevastopol\b'),
    T('Anchorpoint', 'place', r'\bAnchor\s?point\b'),
    T('Thedus', 'place', r'\bThedus\b'),
    T('Origae-6', 'place', r'\bOrigae'),
    T('Gateway Station', 'place', r'\bGateway\b'),
    T('Neverland', 'place', r'\bNeverland\b'),

    # ---------------- 组织 ----------------
    T('Weyland-Yutani', 'org', r'Weyland[\s\-]?Yutani|\bWey[\s\-]?Yu\b'),
    T('Weyland Corp', 'org', r'\bWeyland (?:Corp|Corporation|Industries|Ind\b)'),
    T('Weyland (bare)', 'org', r'\bWeyland\b'),
    T('Yutani', 'org', r'\bYutani\b'),
    T('Seegson', 'org', r'\bSeegson\b'),
    T('USCM/USCMC', 'org', r'(?-i:\bU\.?S\.?C\.?M\.?C?\.?\b)|colonial marine corps'),
    T('UPP', 'org', r'\bUPP\b', cs=True),
    T('Three World Empire', 'org', r'Three[\s\-]?World Empire'),
    T('United Americas', 'org', r'United Americas'),
    T('ICSC', 'org', r'\bICSC\b', cs=True),
    T('Colonial Marshals', 'org', r'Colonial Marshals?|\bMarshals?\b'),
    T('the Company', 'org', r'\bthe Company\b'),
    T('Prodigy', 'org', r'\bProdigy\b'),

    # ---------------- 人物 ----------------
    T('Ripley', 'person', r'\bRipley\b'),
    T('Ellen Ripley', 'person', r'\bEllen\b'),
    T('Amanda Ripley', 'person', r'\bAmanda\b'),
    T('Ash', 'person', r'\bAsh\b', cs=True),
    T('Bishop', 'person', r'\bBishop\b'),
    T('David', 'person', r'\bDavid\b'),
    T('Walter', 'person', r'\bWalter\b'),
    T('Dallas', 'person', r'\bDallas\b'),
    T('Kane', 'person', r'\bKane\b'),
    T('Parker', 'person', r'\bParker\b'),
    T('Lambert', 'person', r'\bLambert\b'),
    T('Brett', 'person', r'\bBrett\b'),
    T('Hicks', 'person', r'\bHicks\b'),
    T('Hudson', 'person', r'\bHudson\b'),
    T('Vasquez', 'person', r'\bVasquez\b'),
    T('Gorman', 'person', r'\bGorman\b'),
    T('Newt', 'person', r'\bNewt\b'),
    T('Burke', 'person', r'\bBurke\b'),
    T('Peter Weyland', 'person', r'Peter Weyland'),
    T('Meredith Vickers', 'person', r'\bVickers\b|\bMeredith\b'),
    T('Shaw', 'person', r'\bShaw\b'),
    T('Rain', 'person', r'\bRain\b'),
    T('Andy', 'person', r'\bAndy\b'),
    T('MU/TH/UR (Mother)', 'person', r'MU[\s/\-]?TH[\s/\-]?UR|\bMother\b'),

    # ---------------- 生物 ----------------
    T('xenomorph', 'creature', r'\bxeno'),
    T('alien (any)', 'creature', r'\bAliens?\b'),
    T('facehugger', 'creature', r'face[\s\-]?hugger'),
    T('chestburster', 'creature', r'chest[\s\-]?burst'),
    T('egg', 'creature', r'\beggs?\b'),
    T('ovomorph', 'creature', r'\bovomorph'),
    T('queen', 'creature', r'\bqueen\b'),
    T('drone', 'creature', r'\bdrones?\b'),
    T('warrior', 'creature', r'\bwarriors?\b'),
    T('neomorph', 'creature', r'\bneomorph'),
    T('host', 'creature', r'\bhosts?\b'),
    T('parasite', 'creature', r'\bparasit'),
    T('organism', 'creature', r'\borganisms?\b'),
    T('specimen', 'creature', r'\bspecimens?\b'),
    T('acid', 'creature', r'\bacids?\b|\bacidic\b'),
    T('hive', 'creature', r'\bhive\b'),
    T('nest', 'creature', r'\bnests?\b'),
    T('cocoon', 'creature', r'\bcocoon'),
    T('Engineer', 'creature', r'\bEngineers?\b'),
    T('creature', 'creature', r'\bcreatures?\b'),
    T('bioweapon', 'creature', r'bio[\s\-]?weapon|biological weapon'),
    T('contagion/infection', 'creature', r'\bcontagion\b|\binfect'),
    T('quarantine', 'creature', r'\bquarantine'),

    # ---------------- 装备 ----------------
    T('pulse rifle', 'hardware', r'pulse rifle'),
    T('smartgun', 'hardware', r'smart[\s\-]?gun'),
    T('motion tracker', 'hardware', r'motion\s*(?:tracker|sensor|detector)'),
    T('tracker (loose)', 'hardware', r'\btrackers?\b'),
    T('flamethrower', 'hardware', r'flame[\s\-]?throw|\bflamer\b|\bflame unit\b'),
    T('power loader', 'hardware', r'power[\s\-]?loader|\bloaders?\b'),
    T('dropship', 'hardware', r'drop[\s\-]?ship'),
    T('APC', 'hardware', r'(?-i:\bA\.?P\.?C\.?\b)|armou?red personnel carrier'),
    T('incinerator', 'hardware', r'\bincinerat'),
    T('sentry gun', 'hardware', r'\bsentry\b'),
    T('airlock', 'hardware', r'air[\s\-]?lock'),
    T('hypersleep', 'hardware', r'hyper[\s\-]?sleep'),
    T('cryo', 'hardware', r'\bcryo'),
    T('stasis', 'hardware', r'\bstasis\b'),
    T('EEV', 'hardware', r'(?-i:\bE\.?E\.?V\.?\b)|emergency escape vehicle'),
    T('lifeboat', 'hardware', r'life[\s\-]?boat|escape (?:pod|shuttle|craft)|\bshuttle\b'),
    T('atmosphere processor', 'hardware', r'atmospher\w* processor|processing station|\bprocessor\b'),
    T('terraforming', 'hardware', r'terraform'),
    T('self-destruct', 'hardware', r'self[\s\-]?destruct'),

    # ---------------- 身份 / 军衔 ----------------
    T('synthetic', 'role', r'\bsynthetics?\b'),
    T('android', 'role', r'\bandroids?\b'),
    T('robot', 'role', r'\brobots?\b'),
    T('artificial person', 'role', r'artificial (?:person|human|being)'),
    T('Colonial Marine', 'role', r'colonial marines?'),
    T('marine (loose)', 'role', r'\bmarines?\b'),
    T('Corporal', 'role', r'\bcorporal\b'),
    T('Sergeant', 'role', r'\bsergeant\b|\bsarge\b'),
    T('Lieutenant', 'role', r'\blieutenant\b'),
    T('Captain', 'role', r'\bcaptain\b'),
    T('Warrant Officer', 'role', r'warrant officer'),
    T('Executive Officer', 'role', r'executive officer'),
    T('Science Officer', 'role', r'science officer'),
    T('science division', 'role', r'science division'),
    T('colonist', 'role', r'\bcolonists?\b|\bcolony\b|\bcolonies\b'),
    T('crew', 'role', r'\bcrew\b'),
]

# 人工审定候选：精确计票（0 也照报——「新血脉根本没有这个词」本身就是结论）
PROBES = {
 "Nostromo": [
  "诺斯都罗莫",
  "诺史莫",
  "诺斯特罗莫",
  "诺斯托罗莫"
 ],
 "Sulaco": [
  "苏拉柯",
  "苏拉科",
  "苏拉可",
  "苏拉"
 ],
 "Covenant": [
  "契约号",
  "圣约号",
  "盟约"
 ],
 "Prometheus": [
  "普罗米修斯",
  "普罗米休斯"
 ],
 "Corbelan": [
  "科布伦",
  "柯贝兰"
 ],
 "Cronus": [
  "克洛诺斯",
  "克罗诺斯"
 ],
 "Romulus": [
  "罗穆路斯",
  "罗慕路斯",
  "罗穆卢斯"
 ],
 "Remus": [
  "雷穆斯",
  "雷姆斯",
  "瑞摩斯"
 ],
 "Narcissus": [
  "水仙",
  "纳西瑟斯"
 ],
 "Betty": [
  "贝蒂"
 ],
 "LV-426": [
  "LV426",
  "LV-426"
 ],
 "LV-223": [
  "LV223",
  "LV-223"
 ],
 "Acheron": [
  "阿契隆",
  "阿刻戎",
  "冥河"
 ],
 "Hadley's Hope": [
  "哈德利",
  "哈德里",
  "希望镇"
 ],
 "Fiorina 161": [
  "菲奥里纳",
  "复仇",
  "161"
 ],
 "Sevastopol": [
  "塞瓦斯托波尔",
  "塞瓦斯托波"
 ],
 "Anchorpoint": [
  "锚点",
  "安克角"
 ],
 "Thedus": [
  "西德斯",
  "塞杜斯"
 ],
 "Origae-6": [
  "欧瑞伽",
  "奥利基",
  "奥瑞加"
 ],
 "Gateway Station": [
  "门户",
  "关口",
  "基地"
 ],
 "Neverland": [
  "梦幻岛",
  "永无岛"
 ],
 "Weyland-Yutani": [
  "威兰汤谷",
  "维兰德",
  "威兰",
  "汤谷",
  "韦兰",
  "尤塔尼"
 ],
 "Weyland Corp": [
  "维兰德公司",
  "维兰德工业",
  "威兰公司",
  "维兰德"
 ],
 "Weyland (bare)": [
  "维兰德",
  "威兰",
  "韦兰"
 ],
 "Yutani": [
  "汤谷",
  "尤塔尼",
  "宇塔尼"
 ],
 "Seegson": [
  "西格森",
  "希格森"
 ],
 "USCM/USCMC": [
  "殖民地陆战队",
  "陆战队",
  "海军陆战队"
 ],
 "UPP": [
  "人民进步联盟",
  "进步人民联盟"
 ],
 "Three World Empire": [
  "三星帝国",
  "三世界帝国"
 ],
 "United Americas": [
  "美洲联合",
  "联合美洲"
 ],
 "ICSC": [
  "星际商业",
  "行星际"
 ],
 "Colonial Marshals": [
  "殖民地执法官",
  "警长",
  "法警"
 ],
 "the Company": [
  "公司",
  "企业"
 ],
 "Prodigy": [
  "天才",
  "神童"
 ],
 "Ripley": [
  "蕾普丽",
  "蕾普莉",
  "里普利",
  "雷普利",
  "蕾普利"
 ],
 "Ellen Ripley": [
  "海伦",
  "爱伦",
  "艾伦",
  "埃伦"
 ],
 "Amanda Ripley": [
  "阿曼达",
  "亚曼达",
  "蕾普丽-麦克伦"
 ],
 "Ash": [
  "艾希",
  "艾许",
  "阿什"
 ],
 "Bishop": [
  "主教",
  "毕晓普",
  "毕夏普"
 ],
 "David": [
  "大卫",
  "戴维"
 ],
 "Walter": [
  "沃尔特",
  "华特"
 ],
 "Dallas": [
  "达拉斯"
 ],
 "Kane": [
  "肯恩",
  "凯恩"
 ],
 "Parker": [
  "派克",
  "帕克"
 ],
 "Lambert": [
  "兰波特",
  "兰伯特"
 ],
 "Brett": [
  "布雷特",
  "布瑞特"
 ],
 "Hicks": [
  "希克斯"
 ],
 "Hudson": [
  "哈德逊",
  "哈德森",
  "哈德孙"
 ],
 "Vasquez": [
  "娃丝佳",
  "瓦斯奎兹",
  "巴斯奎兹"
 ],
 "Gorman": [
  "高曼",
  "戈尔曼"
 ],
 "Newt": [
  "纽特",
  "妞特"
 ],
 "Burke": [
  "巴克",
  "伯克"
 ],
 "Peter Weyland": [
  "彼得",
  "维兰德"
 ],
 "Meredith Vickers": [
  "维克斯",
  "梅雷迪思",
  "梅瑞迪斯"
 ],
 "Shaw": [
  "肖",
  "萧"
 ],
 "Rain": [
  "小雨",
  "蕾恩"
 ],
 "Andy": [
  "安迪"
 ],
 "MU/TH/UR (Mother)": [
  "母亲",
  "老妈",
  "妈妈"
 ],
 "xenomorph": [
  "异形",
  "异型",
  "异种"
 ],
 "alien (any)": [
  "异形",
  "外星"
 ],
 "facehugger": [
  "抱脸虫",
  "抱脸",
  "面部拥抱者"
 ],
 "chestburster": [
  "破胸",
  "爆胸",
  "胸口"
 ],
 "egg": [
  "卵",
  "蛋"
 ],
 "ovomorph": [
  "卵",
  "卵形体"
 ],
 "queen": [
  "女王",
  "王后",
  "蜂后",
  "母后"
 ],
 "drone": [
  "工蜂",
  "无人机",
  "雄虫"
 ],
 "warrior": [
  "战士",
  "武士",
  "勇士"
 ],
 "neomorph": [
  "新形",
  "新异形"
 ],
 "host": [
  "宿主",
  "寄主"
 ],
 "parasite": [
  "寄生虫",
  "寄生物",
  "寄生体",
  "寄生"
 ],
 "organism": [
  "生物",
  "有机体",
  "生命体",
  "生物体"
 ],
 "specimen": [
  "标本",
  "样本",
  "样品"
 ],
 "acid": [
  "强酸",
  "酸性",
  "酸液",
  "酸血",
  "酸"
 ],
 "hive": [
  "巢穴",
  "蜂巢",
  "虫巢",
  "蚂蚁窝"
 ],
 "nest": [
  "巢穴",
  "筑巢",
  "窝"
 ],
 "cocoon": [
  "茧",
  "卵状"
 ],
 "Engineer": [
  "工程师"
 ],
 "creature": [
  "生物",
  "怪物",
  "东西"
 ],
 "bioweapon": [
  "生化武器",
  "生物武器"
 ],
 "contagion/infection": [
  "感染",
  "传染",
  "传染病"
 ],
 "quarantine": [
  "隔离",
  "检疫"
 ],
 "pulse rifle": [
  "脉冲步枪",
  "电波枪",
  "电波机关枪",
  "电波散弹枪",
  "电波"
 ],
 "smartgun": [
  "机关枪",
  "智能枪",
  "智慧型机枪"
 ],
 "motion tracker": [
  "行动追踪器",
  "运动追踪器",
  "动作侦测器",
  "追踪器"
 ],
 "tracker (loose)": [
  "追踪器",
  "追踪"
 ],
 "flamethrower": [
  "喷火枪",
  "火焰喷射器",
  "喷火器"
 ],
 "power loader": [
  "起重机械人",
  "动力装载机",
  "装载机"
 ],
 "dropship": [
  "运输艇",
  "登陆艇",
  "投放船",
  "一艘船"
 ],
 "APC": [
  "焚化炉",
  "装甲运兵车",
  "运兵车",
  "装甲车"
 ],
 "incinerator": [
  "焚化炉",
  "喷火枪",
  "焚化"
 ],
 "sentry gun": [
  "自动机枪",
  "哨戒枪",
  "岗哨",
  "遥控"
 ],
 "airlock": [
  "气闸",
  "气闸舱",
  "空气舱",
  "气舱",
  "空气阀",
  "气锁室",
  "气锁",
  "气密门"
 ],
 "hypersleep": [
  "长眠",
  "冬眠",
  "超级睡眠",
  "休眠"
 ],
 "cryo": [
  "低温舱",
  "低温",
  "冷冻",
  "冻眠",
  "冷冻舱",
  "休眠"
 ],
 "stasis": [
  "静态平衡",
  "静滞",
  "停滞"
 ],
 "EEV": [
  "紧急逃生艇",
  "逃生艇",
  "救生艇"
 ],
 "lifeboat": [
  "救生艇",
  "逃生艇",
  "穿梭机",
  "太空梭"
 ],
 "atmosphere processor": [
  "大气处理厂",
  "大气处理",
  "大气"
 ],
 "terraforming": [
  "地球化",
  "整地",
  "改造"
 ],
 "self-destruct": [
  "自毁",
  "自动引爆",
  "自我毁灭"
 ],
 "synthetic": [
  "合成人",
  "人造人",
  "仿生人",
  "合成"
 ],
 "android": [
  "生化人",
  "机器人",
  "人造人",
  "仿生人",
  "机械人"
 ],
 "robot": [
  "机器人",
  "机械人"
 ],
 "artificial person": [
  "人造人",
  "人工人"
 ],
 "Colonial Marine": [
  "殖民地陆战队",
  "殖民陆战队",
  "殖民地海军陆战队"
 ],
 "marine (loose)": [
  "陆战队",
  "海军陆战队"
 ],
 "Corporal": [
  "下士",
  "伍长"
 ],
 "Sergeant": [
  "中士",
  "士官长",
  "军士"
 ],
 "Lieutenant": [
  "中尉",
  "上尉",
  "少尉"
 ],
 "Captain": [
  "船长",
  "舰长",
  "上尉",
  "机长"
 ],
 "Warrant Officer": [
  "准尉",
  "军官",
  "士官"
 ],
 "Executive Officer": [
  "大副",
  "副长",
  "执行官"
 ],
 "Science Officer": [
  "科学官",
  "科学研究员",
  "首席科学家",
  "科学主任"
 ],
 "science division": [
  "卫生署",
  "科学部",
  "科研部"
 ],
 "colonist": [
  "殖民地",
  "殖民者",
  "移民",
  "拓荒者"
 ],
 "crew": [
  "船员",
  "组员",
  "人员",
  "机组"
 ]
}
for _t in TERMS:
    if _t['key'] in PROBES:
        _t['zh'] = PROBES[_t['key']]
