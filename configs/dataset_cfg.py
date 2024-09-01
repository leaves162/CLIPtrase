dataset_info = {
    "ADE150":{
        "image_path":"your_dataset/ADEChallengeData2016/images/validation",
        "gt_path":"your_dataset/ADEChallengeData2016/annotations_detectron2/validation",
        "labels":["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"]
    },
    "ADEfull":{
        "image_path":"your_dataset/ADE20K_2021_17_01/images_detectron2/validation",
        "gt_path":"your_dataset/ADE20K_2021_17_01/annotations_detectron2/validation",
        "labels":['wall', 'building, edifice', 'sky', 'tree', 'road, route', 'floor, flooring', 'ceiling', 'bed', 'sidewalk, pavement', 'earth, ground', 'cabinet', 'person, individual, someone, somebody, mortal, soul', 'grass', 'windowpane, window', 'car, auto, automobile, machine, motorcar', 'mountain, mount', 'plant, flora, plant life', 'table', 'chair', 'curtain, drape, drapery, mantle, pall', 'door', 'sofa, couch, lounge', 'sea', 'painting, picture', 'water', 'mirror', 'house', 'rug, carpet, carpeting', 'shelf', 'armchair', 'fence, fencing', 'field', 'lamp', 'rock, stone', 'seat', 'river', 'desk', 'bathtub, bathing tub, bath, tub', 'railing, rail', 'signboard, sign', 'cushion', 'path', 'work surface', 'stairs, steps', 'column, pillar', 'sink', 'wardrobe, closet, press', 'snow', 'refrigerator, icebox', 'base, pedestal, stand', 'bridge, span', 'blind, screen', 'runway', 'cliff, drop, drop-off', 'sand', 'fireplace, hearth, open fireplace', 'pillow', 'screen door, screen', 'toilet, can, commode, crapper, pot, potty, stool, throne', 'skyscraper', 'grandstand, covered stand', 'box', 'pool table, billiard table, snooker table', 'palm, palm tree', 'double door', 'coffee table, cocktail table', 'counter', 'countertop', 'chest of drawers, chest, bureau, dresser', 'kitchen island', 'boat', 'waterfall, falls', 'stove, kitchen stove, range, kitchen range, cooking stove', 'flower', 'bookcase', 'controls', 'book', 'stairway, staircase', 'streetlight, street lamp', 'computer, computing machine, computing device, data processor, electronic computer, information processing system', 'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle', 'swivel chair', 'light, light source', 'bench', 'case, display case, showcase, vitrine', 'towel', 'fountain', 'embankment', 'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box', 'van', 'hill', 'awning, sunshade, sunblind', 'poster, posting, placard, notice, bill, card', 'truck, motortruck', 'airplane, aeroplane, plane', 'pole', 'tower', 'court', 'ball', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'buffet, counter, sideboard', 'hovel, hut, hutch, shack, shanty', 'apparel, wearing apparel, dress, clothes', 'minibike, motorbike', 'animal, animate being, beast, brute, creature, fauna', 'chandelier, pendant, pendent', 'step, stair', 'booth, cubicle, stall, kiosk', 'bicycle, bike, wheel, cycle', 'doorframe, doorcase', 'sconce', 'pond', 'trade name, brand name, brand, marque', 'bannister, banister, balustrade, balusters, handrail', 'bag', 'traffic light, traffic signal, stoplight', 'gazebo', 'escalator, moving staircase, moving stairway', 'land, ground, soil', 'board, plank', 'arcade machine', 'eiderdown, duvet, continental quilt', 'bar', 'stall, stand, sales booth', 'playground', 'ship', 'ottoman, pouf, pouffe, puff, hassock', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'bottle', 'cradle', 'pot, flowerpot', 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'train, railroad train', 'stool', 'lake', 'tank, storage tank', 'ice, water ice', 'basket, handbasket', 'manhole', 'tent, collapsible shelter', 'canopy', 'microwave, microwave oven', 'barrel, cask', 'dirt track', 'beam', 'dishwasher, dish washer, dishwashing machine', 'plate', 'screen, crt screen', 'ruins', 'washer, automatic washer, washing machine', 'blanket, cover', 'plaything, toy', 'food, solid food', 'screen, silver screen, projection screen', 'oven', 'stage', 'beacon, lighthouse, beacon light, pharos', 'umbrella', 'sculpture', 'aqueduct', 'container', 'scaffolding, staging', 'hood, exhaust hood', 'curb, curbing, kerb', 'roller coaster', 'horse, equus caballus', 'catwalk', 'glass, drinking glass', 'vase', 'central reservation', 'carousel', 'radiator', 'closet', 'machine', 'pier, wharf, wharfage, dock', 'fan', 'inflatable bounce game', 'pitch', 'paper', 'arcade, colonnade', 'hot tub', 'helicopter', 'tray', 'partition, divider', 'vineyard', 'bowl', 'bullring', 'flag', 'pot', 'footbridge, overcrossing, pedestrian bridge', 'shower', 'bag, traveling bag, travelling bag, grip, suitcase', 'bulletin board, notice board', 'confessional booth', 'trunk, tree trunk, bole', 'forest', 'elevator door', 'laptop, laptop computer', 'instrument panel', 'bucket, pail', 'tapestry, tapis', 'platform', 'jacket', 'gate', 'monitor, monitoring device', 'telephone booth, phone booth, call box, telephone box, telephone kiosk', 'spotlight, spot', 'ring', 'control panel', 'blackboard, chalkboard', 'air conditioner, air conditioning', 'chest', 'clock', 'sand dune', 'pipe, pipage, piping', 'vault', 'table football', 'cannon', 'swimming pool, swimming bath, natatorium', 'fluorescent, fluorescent fixture', 'statue', 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', 'exhibitor', 'ladder', 'carport', 'dam', 'pulpit', 'skylight, fanlight', 'water tower', 'grill, grille, grillwork', 'display board', 'pane, pane of glass, window glass', 'rubbish, trash, scrap', 'ice rink', 'fruit', 'patio', 'vending machine', 'telephone, phone, telephone set', 'net', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'jar', 'track', 'magazine', 'shutter', 'roof', 'banner, streamer', 'landfill', 'post', 'altarpiece, reredos', 'hat, chapeau, lid', 'arch, archway', 'table game', 'bag, handbag, pocketbook, purse', 'document, written document, papers', 'dome', 'pier', 'shanties', 'forecourt', 'crane', 'dog, domestic dog, canis familiaris', 'piano, pianoforte, forte-piano', 'drawing', 'cabin', 'ad, advertisement, advertizement, advertising, advertizing, advert', 'amphitheater, amphitheatre, coliseum', 'monument', 'henhouse', 'cockpit', 'heater, warmer', 'windmill, aerogenerator, wind generator', 'pool', 'elevator, lift', 'decoration, ornament, ornamentation', 'labyrinth', 'text, textual matter', 'printer', 'mezzanine, first balcony', 'mattress', 'straw', 'stalls', 'patio, terrace', 'billboard, hoarding', 'bus stop', 'trouser, pant', 'console table, console', 'rack', 'notebook', 'shrine', 'pantry', 'cart', 'steam shovel', 'porch', 'postbox, mailbox, letter box', 'figurine, statuette', 'recycling bin', 'folding screen', 'telescope', 'deck chair, beach chair', 'kennel', 'coffee maker', "altar, communion table, lord's table", 'fish', 'easel', 'artificial golf green', 'iceberg', 'candlestick, candle holder', 'shower stall, shower bath', 'television stand', 'wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle', 'skeleton', 'grand piano, grand', 'candy, confect', 'grille door', 'pedestal, plinth, footstall', 'jersey, t-shirt, tee shirt', 'shoe', 'gravestone, headstone, tombstone', 'shanty', 'structure', 'rocking chair, rocker', 'bird', 'place mat', 'tomb', 'big top', 'gas pump, gasoline pump, petrol pump, island dispenser', 'lockers', 'cage', 'finger', 'bleachers', 'ferris wheel', 'hairdresser chair', 'mat', 'stands', 'aquarium, fish tank, marine museum', 'streetcar, tram, tramcar, trolley, trolley car', 'napkin, table napkin, serviette', 'dummy', 'booklet, brochure, folder, leaflet, pamphlet', 'sand trap', 'shop, store', 'table cloth', 'service station', 'coffin', 'drawer', 'cages', 'slot machine, coin machine', 'balcony', 'volleyball court', 'table tennis', 'control table', 'shirt', 'merchandise, ware, product', 'railway', 'parterre', 'chimney', 'can, tin, tin can', 'tanks', 'fabric, cloth, material, textile', 'alga, algae', 'system', 'map', 'greenhouse', 'mug', 'barbecue', 'trailer', 'toilet tissue, toilet paper, bathroom tissue', 'organ', 'dishrag, dishcloth', 'island', 'keyboard', 'trench', 'basket, basketball hoop, hoop', 'steering wheel, wheel', 'pitcher, ewer', 'goal', 'bread, breadstuff, staff of life', 'beds', 'wood', 'file cabinet', 'newspaper, paper', 'motorboat', 'rope', 'guitar', 'rubble', 'scarf', 'barrels', 'cap', 'leaves', 'control tower', 'dashboard', 'bandstand', 'lectern', 'switch, electric switch, electrical switch', 'baseboard, mopboard, skirting board', 'shower room', 'smoke', 'faucet, spigot', 'bulldozer', 'saucepan', 'shops', 'meter', 'crevasse', 'gear', 'candelabrum, candelabra', 'sofa bed', 'tunnel', 'pallet', 'wire, conducting wire', 'kettle, boiler', 'bidet', 'baby buggy, baby carriage, carriage, perambulator, pram, stroller, go-cart, pushchair, pusher', 'music stand', 'pipe, tube', 'cup', 'parking meter', 'ice hockey rink', 'shelter', 'weeds', 'temple', 'patty, cake', 'ski slope', 'panel', 'wallet', 'wheel', 'towel rack, towel horse', 'roundabout', 'canister, cannister, tin', 'rod', 'soap dispenser', 'bell', 'canvas', 'box office, ticket office, ticket booth', 'teacup', 'trellis', 'workbench', 'valley, vale', 'toaster', 'knife', 'podium', 'ramp', 'tumble dryer', 'fireplug, fire hydrant, plug', 'gym shoe, sneaker, tennis shoe', 'lab bench', 'equipment', 'rocky formation', 'plastic', 'calendar', 'caravan', 'check-in-desk', 'ticket counter', 'brush', 'mill', 'covered bridge', 'bowling alley', 'hanger', 'excavator', 'trestle', 'revolving door', 'blast furnace', 'scale, weighing machine', 'projector', 'soap', 'locker', 'tractor', 'stretcher', 'frame', 'grating', 'alembic', 'candle, taper, wax light', 'barrier', 'cardboard', 'cave', 'puddle', 'tarp', 'price tag', 'watchtower', 'meters', 'light bulb, lightbulb, bulb, incandescent lamp, electric light, electric-light bulb', 'tracks', 'hair dryer', 'skirt', 'viaduct', 'paper towel', 'coat', 'sheet', 'fire extinguisher, extinguisher, asphyxiator', 'water wheel', 'pottery, clayware', 'magazine rack', 'teapot', 'microphone, mike', 'support', 'forklift', 'canyon', 'cash register, register', 'leaf, leafage, foliage', 'remote control, remote', 'soap dish', 'windshield, windscreen', 'cat', 'cue, cue stick, pool cue, pool stick', 'vent, venthole, vent-hole, blowhole', 'videos', 'shovel', 'eaves', 'antenna, aerial, transmitting aerial', 'shipyard', 'hen, biddy', 'traffic cone', 'washing machines', 'truck crane', 'cds', 'niche', 'scoreboard', 'briefcase', 'boot', 'sweater, jumper', 'hay', 'pack', 'bottle rack', 'glacier', 'pergola', 'building materials', 'television camera', 'first floor', 'rifle', 'tennis table', 'stadium', 'safety belt', 'cover', 'dish rack', 'synthesizer', 'pumpkin', 'gutter', 'fruit stand', 'ice floe, floe', 'handle, grip, handgrip, hold', 'wheelchair', 'mousepad, mouse mat', 'diploma', 'fairground ride', 'radio', 'hotplate', 'junk', 'wheelbarrow', 'stream', 'toll plaza', 'punching bag', 'trough', 'throne', 'chair desk', 'weighbridge', 'extractor fan', 'hanging clothes', 'dish, dish aerial, dish antenna, saucer', 'alarm clock, alarm', 'ski lift', 'chain', 'garage', 'mechanical shovel', 'wine rack', 'tramway', 'treadmill', 'menu', 'block', 'well', 'witness stand', 'branch', 'duck', 'casserole', 'frying pan', 'desk organizer', 'mast', 'spectacles, specs, eyeglasses, glasses', 'service elevator', 'dollhouse', 'hammock', 'clothes hanging', 'photocopier', 'notepad', 'golf cart', 'footpath', 'cross', 'baptismal font', 'boiler', 'skip', 'rotisserie', 'tables', 'water mill', 'helmet', 'cover curtain', 'brick', 'table runner', 'ashtray', 'street box', 'stick', 'hangers', 'cells', 'urinal', 'centerpiece', 'portable fridge', 'dvds', 'golf club', 'skirting board', 'water cooler', 'clipboard', 'camera, photographic camera', 'pigeonhole', 'chips', 'food processor', 'post box', 'lid', 'drum', 'blender', 'cave entrance', 'dental chair', 'obelisk', 'canoe', 'mobile', 'monitors', 'pool ball', 'cue rack', 'baggage carts', 'shore', 'fork', 'paper filer', 'bicycle rack', 'coat rack', 'garland', 'sports bag', 'fish tank', 'towel dispenser', 'carriage', 'brochure', 'plaque', 'stringer', 'iron', 'spoon', 'flag pole', 'toilet brush', 'book stand', 'water faucet, water tap, tap, hydrant', 'ticket office', 'broom', 'dvd', 'ice bucket', 'carapace, shell, cuticle, shield', 'tureen', 'folders', 'chess', 'root', 'sewing machine', 'model', 'pen', 'violin', 'sweatshirt', 'recycling materials', 'mitten', 'chopping board, cutting board', 'mask', 'log', 'mouse, computer mouse', 'grill', 'hole', 'target', 'trash bag', 'chalk', 'sticks', 'balloon', 'score', 'hair spray', 'roll', 'runner', 'engine', 'inflatable glove', 'games', 'pallets', 'baskets', 'coop', 'dvd player', 'rocking horse', 'buckets', 'bread rolls', 'shawl', 'watering can', 'spotlights', 'post-it', 'bowls', 'security camera', 'runner cloth', 'lock', 'alarm, warning device, alarm system', 'side', 'roulette', 'bone', 'cutlery', 'pool balls', 'wheels', 'spice rack', 'plant pots', 'towel ring', 'bread box', 'video', 'funfair', 'breads', 'tripod', 'ironing board', 'skimmer', 'hollow', 'scratching post', 'tricycle', 'file box', 'mountain pass', 'tombstones', 'cooker', 'card game, cards', 'golf bag', 'towel paper', 'chaise lounge', 'sun', 'toilet paper holder', 'rake', 'key', 'umbrella stand', 'dartboard', 'transformer', 'fireplace utensils', 'sweatshirts', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'tallboy', 'stapler', 'sauna', 'test tube', 'palette', 'shopping carts', 'tools', 'push button, push, button', 'star', 'roof rack', 'barbed wire', 'spray', 'ear', 'sponge', 'racket', 'tins', 'eyeglasses', 'file', 'scarfs', 'sugar bowl', 'flip flop', 'headstones', 'laptop bag', 'leash', 'climbing frame', 'suit hanger', 'floor spotlight', 'plate rack', 'sewer', 'hard drive', 'sprinkler', 'tools box', 'necklace', 'bulbs', 'steel industry', 'club', 'jack', 'door bars', 'control panel, instrument panel, control board, board, panel', 'hairbrush', 'napkin holder', 'office', 'smoke detector', 'utensils', 'apron', 'scissors', 'terminal', 'grinder', 'entry phone', 'newspaper stand', 'pepper shaker', 'onions', 'central processing unit, cpu, c p u , central processor, processor, mainframe', 'tape', 'bat', 'coaster', 'calculator', 'potatoes', 'luggage rack', 'salt', 'street number', 'viewpoint', 'sword', 'cd', 'rowing machine', 'plug', 'andiron, firedog, dog, dog-iron', 'pepper', 'tongs', 'bonfire', 'dog dish', 'belt', 'dumbbells', 'videocassette recorder, vcr', 'hook', 'envelopes', 'shower faucet', 'watch', 'padlock', 'swimming pool ladder', 'spanners', 'gravy boat', 'notice board', 'trash bags', 'fire alarm', 'ladle', 'stethoscope', 'rocket', 'funnel', 'bowling pins', 'valve', 'thermometer', 'cups', 'spice jar', 'night light', 'soaps', 'games table', 'slotted spoon', 'reel', 'scourer', 'sleeping robe', 'desk mat', 'dumbbell', 'hammer', 'tie', 'typewriter', 'shaker', 'cheese dish', 'sea star', 'racquet', 'butane gas cylinder', 'paper weight', 'shaving brush', 'sunglasses', 'gear shift', 'towel rail', 'adding machine, totalizer, totaliser']
    },
    "COCO80_val":{
        "image_path":"your_dataset/coco/val2017",
        "gt_path":"your_dataset/coco/stuffthingmaps_detectron2/val2017",
        "labels":['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
        "background":['textile','building','furniture','structural','floor','ceiling','sky','ground','water','food','solid','wall','window','desk','lamp','wardrobe','banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'],
    },
    "COCO171_val":{
        "image_path":"your_dataset/coco/val2017",
        "gt_path":"your_dataset/coco/stuffthingmaps_detectron2/val2017",
        "labels":['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood']
    },
    "COCO171_train":{
        "image_path":"your_dataset/coco/train2017",
        "gt_path":"your_dataset/coco/stuffthingmaps_detectron2/train2017",
        "labels":['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood']
    },
    "PC59":{
        "image_path":"your_dataset/pcontext/val/image",
        "gt_path":"your_dataset/pcontext/val/label",
        "labels":["aeroplane","bag","bed","bedclothes","bench","bicycle","bird","boat","book","bottle","building","bus","cabinet","car","cat","ceiling","chair","cloth","computer","cow","cup","curtain","dog","door","fence","floor","flower","food","grass","ground","horse","keyboard","light","motorbike","mountain","mouse","person","plate","platform","pottedplant","road","rock","sheep","shelves","sidewalk","sign","sky","snow","sofa","diningtable","track","train","tree","truck","tvmonitor","wall","water","window","wood"]
    },
    "PC60":{
        "image_path":"your_dataset/pcontext/val/image",
        "gt_path":"your_dataset/pcontext/val/label",
        "labels":["aeroplane","bag","bed","bedclothes","bench","bicycle","bird","boat","book","bottle","building","bus","cabinet","car","cat","ceiling","chair","cloth","computer","cow","cup","curtain","dog","door","fence","floor","flower","food","grass","ground","horse","keyboard","light","motorbike","mountain","mouse","person","plate","platform","pottedplant","road","rock","sheep","shelves","sidewalk","sign","sky","snow","sofa","diningtable","track","train","tree","truck","tvmonitor","wall","water","window","wood"],
        "background":["background"]
    },
    "PC459":{
        "image_path":"your_dataset/pcontext_full/val/image",
        "gt_path":"your_dataset/pcontext_full/val/label",
        "labels":["accordion","aeroplane","air conditioner","antenna","artillery","ashtray","atrium","baby carriage","bag","ball","balloon","bamboo weaving","barrel","baseball bat","basket","basketball backboard","bathtub","bed","bedclothes","beer","bell","bench","bicycle","binoculars","bird","bird cage","bird feeder","bird nest","blackboard","board","boat","bone","book","bottle","bottle opener","bowl","box","bracelet","brick","bridge","broom","brush","bucket","building","bus","cabinet","cabinet door","cage","cake","calculator","calendar","camel","camera","camera lens","can","candle","candle holder","cap","car","card","cart","case","casette recorder","cash register","cat","cd","cd player","ceiling","cell phone","cello","chain","chair","chessboard","chicken","chopstick","clip","clippers","clock","closet","cloth","clothes tree","coffee","coffee machine","comb","computer","concrete","cone","container","control booth","controller","cooker","copying machine","coral","cork","corkscrew","counter","court","cow","crabstick","crane","crate","cross","crutch","cup","curtain","cushion","cutting board","dais","disc","disc case","dishwasher","dock","dog","dolphin","door","drainer","dray","drink dispenser","drinking machine","drop","drug","drum","drum kit","duck","dumbbell","earphone","earrings","egg","electric fan","electric iron","electric pot","electric saw","electronic keyboard","engine","envelope","equipment","escalator","exhibition booth","extinguisher","eyeglass","fan","faucet","fax machine","fence","ferris wheel","fire extinguisher","fire hydrant","fire place","fish","fish tank","fishbowl","fishing net","fishing pole","flag","flagstaff","flame","flashlight","floor","flower","fly","foam","food","footbridge","forceps","fork","forklift","fountain","fox","frame","fridge","frog","fruit","funnel","furnace","game controller","game machine","gas cylinder","gas hood","gas stove","gift box","glass","glass marble","globe","glove","goal","grandstand","grass","gravestone","ground","guardrail","guitar","gun","hammer","hand cart","handle","handrail","hanger","hard disk drive","hat","hay","headphone","heater","helicopter","helmet","holder","hook","horse","horse-drawn carriage","hot-air balloon","hydrovalve","ice","inflator pump","ipod","iron","ironing board","jar","kart","kettle","key","keyboard","kitchen range","kite","knife","knife block","ladder","ladder truck","ladle","laptop","leaves","lid","life buoy","light","light bulb","lighter","line","lion","lobster","lock","machine","mailbox","mannequin","map","mask","mat","match book","mattress","menu","metal","meter box","microphone","microwave","mirror","missile","model","money","monkey","mop","motorbike","mountain","mouse","mouse pad","musical instrument","napkin","net","newspaper","oar","ornament","outlet","oven","oxygen bottle","pack","pan","paper","paper box","paper cutter","parachute","parasol","parterre","patio","pelage","pen","pen container","pencil","person","photo","piano","picture","pig","pillar","pillow","pipe","pitcher","plant","plastic","plate","platform","player","playground","pliers","plume","poker","poker chip","pole","pool table","postcard","poster","pot","pottedplant","printer","projector","pumpkin","rabbit","racket","radiator","radio","rail","rake","ramp","range hood","receiver","recorder","recreational machines","remote control","road","robot","rock","rocket","rocking horse","rope","rug","ruler","runway","saddle","sand","saw","scale","scanner","scissors","scoop","screen","screwdriver","sculpture","scythe","sewer","sewing machine","shed","sheep","shell","shelves","shoe","shopping cart","shovel","sidecar","sidewalk","sign","signal light","sink","skateboard","ski","sky","sled","slippers","smoke","snail","snake","snow","snowmobiles","sofa","spanner","spatula","speaker","speed bump","spice container","spoon","sprayer","squirrel","stage","stair","stapler","stick","sticky note","stone","stool","stove","straw","stretcher","sun","sunglass","sunshade","surveillance camera","swan","sweeper","swim ring","swimming pool","swing","switch","table","tableware","tank","tap","tape","tarp","telephone","telephone booth","tent","tire","toaster","toilet","tong","tool","toothbrush","towel","toy","toy car","track","train","trampoline","trash bin","tray","tree","tricycle","tripod","trophy","truck","tube","turtle","tvmonitor","tweezers","typewriter","umbrella","unknown","vacuum cleaner","vending machine","video camera","video game console","video player","video tape","violin","wakeboard","wall","wallet","wardrobe","washing machine","watch","water","water dispenser","water pipe","water skate board","watermelon","whale","wharf","wheel","wheelchair","window","window blinds","wineglass","wire","wood","wool"]
    },
    "VOC20":{
        "image_path":"your_dataset/VOC2012/images_detectron2/val",
        "gt_path":"your_dataset/VOC2012/annotations_detectron2/val",
        "labels":["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tv"]
    },
    "VOC21":{
        "image_path":"your_dataset/VOC2012/images_detectron2/val",
        "gt_path":"your_dataset/VOC2012/annotations_detectron2/val",
        "labels":["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tv"],
        "background":['background','sky', 'wall','white','tree', 'wood', 'grass', 'road','trail', 'sea', 'river','water','box','signboard', 'mountain', 'sands', 'bed', 'bed', 'building', 'cloud', 'lamp', 'door', 'window', 'wardrobe', 'ceiling', 'shelf', 'curtain', 'stair', 'floor','ground','beach' 'hill', 'rail', 'fence'],
    }
}

prompt_templates = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]
