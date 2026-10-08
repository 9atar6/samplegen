"""Word lists behind Explore and the nudge buttons (prompting.py has the logic).

Stable Audio 3 answers best to plain descriptions: what it is, what it's made of, how big,
how far, in what space. An "angle" puts the user's few words in one such setting; four
contrasting angles give four quite different takes to choose from by ear.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Angle:
    template: str              # "{x}" is the user's sketch, word for word
    avoid: frozenset[str]      # skipped when the sketch says one of these (it would contradict it)


def _angle(template: str, *avoid: str) -> Angle:
    return Angle(template, frozenset(avoid))


SFX_ANGLES = (
    _angle("{x}, close-miked, dry, detailed foley",
           "reverb", "echo", "distant", "far", "hall", "cathedral", "cave", "wet", "outdoor", "outdoors"),
    _angle("huge cinematic {x}, deep low end, long reverb tail",
           "small", "tiny", "soft", "subtle", "quiet", "dry", "close", "delicate", "gentle", "cartoon", "reverb"),
    _angle("{x} in the distance, outdoors, natural ambience", "close", "close-miked", "dry", "near", "intimate"),
    _angle("futuristic sci-fi {x}, synthesized, metallic resonance",
           "vintage", "old", "retro", "natural", "organic", "realistic", "wooden"),
    _angle("{x}, vintage recording, lo-fi, warm tape saturation",
           "sci-fi", "futuristic", "modern", "clean", "hi-fi", "crisp", "digital"),
    _angle("small, delicate {x}, soft and subtle",
           "huge", "big", "massive", "loud", "heavy", "cinematic", "powerful", "aggressive", "giant"),
    _angle("heavy, powerful {x}, punchy and aggressive",
           "soft", "subtle", "delicate", "gentle", "small", "tiny", "quiet", "light"),
    _angle("cartoony {x}, playful, exaggerated, bouncy", "realistic", "dark", "horror", "scary", "cinematic"),
    _angle("{x} in a large empty concrete hall, long natural reverb",
           "dry", "close", "close-miked", "outdoor", "outdoors", "small", "reverb", "hall", "echo"),
    _angle("dark, ominous {x}, low rumble, horror", "bright", "happy", "cartoon", "playful", "cute"),
)

FREE_ANGLES = (
    _angle("{x}, dark evolving drone, slow movement", "bright", "happy", "fast"),
    _angle("{x}, shimmering granular texture, airy and bright", "dark", "muffled"),
    _angle("{x}, warm analog pad, lush and wide", "cold", "harsh", "metallic"),
    _angle("{x}, field recording ambience, natural and detailed", "synthesized", "synth", "sci-fi"),
    _angle("{x}, cold metallic resonances, bowed metal", "warm"),
    _angle("{x}, glitchy, stuttering, digital artifacts", "smooth", "natural", "organic"),
    _angle("{x}, ethereal choir-like tones, cathedral reverb", "dry"),
    _angle("{x}, deep sub rumble, cinematic tension", "light", "airy", "bright"),
)

FREE_SURPRISES = (
    "dark evolving ambient drone, granular, metallic shimmer",
    "warm analog pad, slow filter movement, lush and wide",
    "night forest ambience, crickets, distant owl",
    "underwater cave ambience, deep resonant hum, bubbles",
    "busy city street ambience, traffic, distant voices",
    "glitchy digital texture, stuttering, bitcrushed",
    "ethereal choir drone in a cathedral",
    "spaceship engine room hum, low and steady",
)

# A bare noun the model knows well gets hand-made, contrasting takes on it.
SFX_PACKS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("door", "doors"): (
        "old wooden door creaking open slowly", "heavy metal door slamming shut in a concrete corridor",
        "sci-fi airlock door opening with a pneumatic hiss", "car door closing, solid thunk, close-miked",
        "squeaky screen door banging shut", "huge castle door, deep wooden boom, stone hall reverb"),
    ("footsteps", "footstep", "steps", "walking", "walk"): (
        "footsteps on gravel, slow walk, close-miked", "high heels on a marble floor, echoing hallway",
        "heavy boots running on a metal catwalk", "footsteps crunching in fresh snow",
        "barefoot steps on creaky wooden floorboards"),
    ("impact", "hit", "punch", "thud"): (
        "deep cinematic impact, sub boom, long tail", "punchy body hit, close, dry",
        "metal pipe hitting concrete, ringing resonance", "heavy wooden thud on a table",
        "sci-fi energy impact with an electric crackle"),
    ("whoosh", "swoosh", "swish", "swipe"): (
        "fast air whoosh passing by, left to right", "deep heavy whoosh, slow and cinematic",
        "sharp sword swish, thin and quick", "sci-fi laser whoosh with a pitch drop", "fire whoosh, flame burst"),
    ("explosion", "boom", "blast"): (
        "huge explosion, deep sub boom, debris falling", "distant explosion rumbling across a valley",
        "small firecracker pop, sharp and dry", "sci-fi plasma explosion, electric crackle",
        "underwater explosion, muffled boom and bubbles"),
    ("water", "splash"): (
        "big splash into a lake, close", "water dripping in a cave, echo", "stream babbling over rocks, gentle",
        "pouring water into a glass", "underwater bubbles rising"),
    ("rain",): (
        "soft rain on a tent", "heavy rain on a tin roof", "rain on a car windshield, interior",
        "light rain on leaves in a forest", "thunderstorm downpour with distant thunder"),
    ("wind",): (
        "howling wind through a broken window", "gentle breeze through tall grass", "icy mountain wind gusts",
        "wind whistling through a narrow alley", "desert wind with blowing sand"),
    ("fire", "flame", "flames"): (
        "crackling campfire, close and warm", "roaring house fire, intense flames", "match strike and ignite",
        "fire whoosh, flamethrower burst", "fireplace embers popping softly"),
    ("glass",): (
        "glass bottle shattering on concrete", "wine glasses clinking, cheers", "window pane breaking, debris falling",
        "glass marbles rolling on a wooden floor", "finger rubbing a wine glass rim, ringing tone"),
    ("ui", "click", "button", "notification", "beep"): (
        "soft UI click, clean and short", "bright notification chime, two notes", "sci-fi interface beep, digital",
        "wooden button click, tactile", "error buzz, short and low"),
    ("magic", "spell"): (
        "magic spell cast, shimmering sparkles", "dark magic whoosh with a low drone",
        "healing spell, warm bell tones rising", "ice spell, crystalline crackle",
        "teleport, swirling sweep and pop"),
    ("monster", "creature", "growl", "roar"): (
        "huge monster roar, deep and wet", "small creature chirping, cute", "zombie groan, raspy and slow",
        "dragon growl with fiery breath", "alien creature clicks and hisses"),
    ("riser", "rise", "build", "tension"): (
        "cinematic riser, building tension, ends abruptly", "white noise sweep riser, bright",
        "dark orchestral riser with deep brass", "sci-fi pitch riser, synthesized", "reverse cymbal swell"),
    ("engine", "car", "vehicle", "motor"): (
        "sports car engine revving, passing by", "old truck engine idling, rattling", "electric car humming past",
        "motorcycle starting and pulling away", "spaceship engine hum, low and steady"),
    ("thunder",): (
        "distant thunder rumble", "close thunder crack, sharp and loud", "rolling thunder across mountains",
        "thunderstorm with heavy rain"),
    ("laser",): (
        "sci-fi laser shot, short zap", "laser beam charging up and firing", "retro arcade laser, 8-bit",
        "heavy plasma cannon blast"),
    ("coin", "coins"): (
        "coin pickup, bright chime", "coins dropping on a wooden table", "pile of gold coins pouring",
        "single coin spinning and settling"),
}

# Loop mode: tags Explore adds when the sketch doesn't pick them (Foundation-1 vocabulary).
LOOP_FX = ("Dry", "Low Reverb", "Medium Reverb", "High Reverb", "Medium Delay", "Low Distortion", "Phaser")
LOOP_BASS_STRUCTURES = ("Bassline", "Rolling", "Simple", "Staccato", "Choppy")
LOOP_STRUCTURES = ("Melody", "Chord Progression", "Arp", "Complex Arp Melody", "Sustained", "Simple", "Rising")
# Timbre tags that name a waveform or a trick rather than a character: never added at random.
LOOP_TECHNICAL_TIMBRES = frozenset({"Pitch Bend", "Formant Vocal", "808", "303", "Saw", "Square", "Sine", "Pulse"})
# Timbre tags that can't sit in one prompt.
LOOP_OPPOSITES = (
    {"Dark", "Bright"}, {"Dark", "Sparkly"}, {"Dark", "Shiny"}, {"Dark", "Glassy"}, {"Muffled", "Crisp"},
    {"Muffled", "Bright"}, {"Warm", "Cold"}, {"Thick", "Thin"}, {"Fat", "Thin"}, {"Big", "Small"},
    {"Big", "Intimate"}, {"Clean", "Gritty"}, {"Clean", "Overdriven"}, {"Clean", "Bitcrushed"},
    {"Smooth", "Gritty"}, {"Smooth", "Harsh"}, {"Silky", "Harsh"}, {"Soft", "Harsh"}, {"Soft", "Growl"},
    {"Near", "Distant"}, {"Intimate", "Distant"}, {"Wide", "Focused"}, {"Heavy", "Thin"},
)


@dataclass(frozen=True)
class Nudge:
    label: str
    words: str                 # added to a described sound (SFX / Free)
    tags: str                  # added to a tag prompt (Loop)
    against: frozenset[str]    # words it replaces


def _nudge(label: str, words: str, tags: str, *against: str) -> Nudge:
    return Nudge(label, words, tags, frozenset(against))


NUDGES: dict[str, Nudge] = {
    "darker": _nudge("Darker", "dark, warm, muffled", "Dark, Warm",
                     "bright", "crisp", "airy", "shiny", "sparkly", "glassy", "brilliant"),
    "brighter": _nudge("Brighter", "bright, crisp, airy", "Bright, Crisp", "dark", "muffled", "warm", "subdued", "dull"),
    "bigger": _nudge("Bigger", "huge, wide, deep low end", "Big, Wide, Deep", "small", "thin", "tiny", "tight", "subtle"),
    "tighter": _nudge("Tighter", "short, tight, punchy", "Tight, Punchy", "long", "huge", "big", "sustained", "wide"),
    "drier": _nudge("Drier", "dry, close-miked", "Dry",
                    "reverb", "wet", "echo", "hall", "distant", "delay", "cathedral", "cave"),
    "roomier": _nudge("More space", "long reverb tail, large hall", "High Reverb",
                      "dry", "close-miked", "close", "near", "intimate"),
    "grittier": _nudge("Grittier", "gritty, distorted, saturated", "Gritty, Medium Distortion",
                       "clean", "smooth", "silky", "pristine"),
    "cleaner": _nudge("Cleaner", "clean, smooth", "Clean, Smooth", "gritty", "distorted", "distortion",
                      "bitcrushed", "bitcrush", "lo-fi", "saturated", "overdriven", "dirty"),
}
