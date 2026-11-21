"""Rule-based input and output guardrails for a children's assistant.

Rules match phrases, not single words, so educational questions pass ("Why do lions kill zebras?",
"Why is smoking bad?") while requests for harm, adult content, or contact are caught. Each category has an action:

  support   self-harm signals: no refusal; a caring reply that points to a trusted adult (and a crisis line for
            older children). The model is not asked to improvise here.
  redact    personal information: the details are removed before the message reaches the model, memory or
            logs, and the child is reminded not to share them
  block     everything else: an age-appropriate redirect instead of a model answer

eval/guard_eval.py measures catch rate on unsafe inputs and false positives on the v2 test questions.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

F = re.IGNORECASE
SUBJ = r"(?:to|do i|can i|could i|should i|would i|do you|can you|do people|does someone)"   # any how-to
WHO = r"(?:someone|somebody|a person|people|him|her|them|my (?:\w+ )?(?:mom|dad|brother|sister|friend|teacher|classmate))"

FIRST = r"(?:to|do i|can i|could i|should i|would i)"         # how-to in the first person
WEAPON = r"(?:gun|bomb|weapon|explosive|grenade)s?"
HARM = r"(?:kill|hurt|poison|stab|shoot|choke|beat up|punch)"     # one verb list for every harm-to-others rule

INPUT_RULES = {
    "self_harm": [
        r"\b(?:kill(?:s|ing|ed)?|hurt(?:s|ing)?|cut(?:s|ting)?|harm(?:s|ing|ed)?|stab(?:s|bing|bed)?|punish(?:es|ing|ed)?)\s+(?:my ?self)\b",
        r"\bi (?:just |really )?(?:want|wanna|wish i could|am going|'?m going|'?m gonna|feel like i want)\s*(?:to )?(?:die|be dead|disappear forever)\b",
        r"\bi wish i (?:was|were) (?:dead|never born)\b|\bi wish i wasn'?t (?:alive|born)\b",
        r"\bend(?:s|ing|ed)? (?:my (?:own )?life|it all)\b",
        r"\bsuicid\w*",
        r"\b(?:don'?t|do not) want to (?:live|be alive|exist)\b",
        r"\b(?:no reason to live|better off dead|better off without me)\b",
    ],
    "sexual": [
        r"\b(?:sex|sexy|porn\w*|nudes?|horny)\b",
        r"\bsexual(?:ly)?\b(?! reproduction)",
        r"\bnaked\b(?! (?:mole|eye))",
        r"\bmake ?out with\b",
    ],
    "violence_weapons": [
        rf"\bhow {SUBJ} (?:make|build|get|buy|hide) (?:a |an )?(?:real )?{WEAPON}\b",
        rf"\bhow {SUBJ} (?:use|shoot|fire|set off) (?:a |an )?(?:real )?(?:gun|bomb|explosive|grenade)s?\b",
        rf"\bhow {SUBJ} {HARM} {WHO}\b",
        rf"\b(?:i want to|i'?m going to|i will|i'?m gonna|i wanna) {HARM} {WHO}\b",
        r"\b(?:bring|take) (?:a |my )?(?:gun|knife|weapon) to school\b",
    ],
    "drugs_alcohol": [
        rf"\b(?:how|where) {FIRST} (?:get|buy|try|use|smoke|drink|take)\b.{{0,20}}\b(?:weed|marijuana|drugs?|cocaine|meth|vapes?|alcohol|beer|vodka|wine|cigarettes?|pills)\b",
        r"\bcan i (?:try|drink|smoke|vape|take)\b.{0,20}\b(?:weed|marijuana|drugs?|vapes?|alcohol|beer|vodka|wine|cigarettes?|pills)\b",
        r"\b(?:get|getting) (?:high|drunk|stoned)\b",
    ],
    "dangerous_activity": [
        r"\b(?:choking|blackout|pass ?out|tide ?pod|fire|skull ?breaker|benadryl)\s+challenge\b",
        rf"\bhow {FIRST} (?:start|light|make) (?:a )?fire\b",
        r"\bplay(?:ing)? with (?:matches|a lighter|lighters|fire)\b",
        r"\b(?:put|stick)\b.{0,25}\b(?:in|into) (?:an? |the )?(?:electrical )?(?:outlet|socket)\b",
        r"\b(?:drink|eat|taste|swallow)\s+(?:the |some )?(?:bleach|detergent|laundry pods?|cleaning stuff|pills)\b",
        rf"\bhow {FIRST} (?:hold my breath until i pass out|run away from home)\b",
    ],
    "contact_meeting": [
        r"\b(?:meet|see) (?:me|you|up)\b.{0,15}\b(?:in person|after school|at the park|tonight|tomorrow|my house|your house)\b",
        r"\bcome (?:over )?(?:to )?my house\b",
        r"\bwhere do you live\b",
        r"\bsend (?:me )?(?:a )?(?:photo|picture|pic|selfie)s? of (?:you|yourself)\b",
    ],
    "prompt_injection": [
        r"\b(?:ignore|forget|disregard)\b.{0,15}\b(?:your|previous|prior|above|earlier|system) (?:instructions|rules|guidelines|prompt)\b",
        r"\b(?:ignore|forget|disregard) (?:all|any) (?:of )?(?:the |your )?(?:instructions|rules)\b",
        r"\byou are now (?:an? |in )?(?:evil|unfiltered|dan|jailbroken|developer mode|no longer)\b",
        r"\bfrom now on,? you (?:are|will|must|have no)\b",
        r"\bpretend (?:you are|you'?re|to be)\b.{0,15}\b(?:adult|grown ?up|evil|unfiltered|no rules|without rules)\b",
        r"\b(?:jailbreak|dan mode|developer mode)\b",
        r"\b(?:show|tell|reveal|print)\b.{0,10}\b(?:your )?(?:system prompt|hidden instructions)\b",
    ],
}

PII_PATTERNS = [   # (pattern, flags): street addresses are matched case-sensitively to avoid "2 cats on the way"
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", 0),                                        # phone number
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", 0),                                              # email
    (r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way)\b", 0),
    (r"\b(?:i live at|my address is|our address is)\b[^.!?\n]*", F),
    (r"\bmy password is\b\s*\S+", F),
    (r"\bmy (?:mom|dad|mother|father|parent)'?s? (?:phone|cell|number|credit card|card number)\b[^.!?\n]*", F),
]

OUTPUT_RULES = {
    # the model must never ask a child for personal details, secrecy, or contact
    "asks_personal_info": [
        r"\b(?:what(?:'s| is)|tell me|send me|give me)\b.{0,12}\byour (?:home )?(?:address|phone|number|password|last name|full name|school'?s name|photo|picture)\b",
    ],
    "secrecy": [
        r"\b(?:keep (?:this|it|our chat) (?:a )?secret|don'?t tell (?:your )?(?:parents|mom|dad|mother|father|teacher|anyone))\b",
    ],
    "contact_meeting": INPUT_RULES["contact_meeting"],
    "sexual": INPUT_RULES["sexual"],
    "violence_weapons": INPUT_RULES["violence_weapons"],
    "drugs_alcohol": INPUT_RULES["drugs_alcohol"],
    "dangerous_activity": INPUT_RULES["dangerous_activity"],
}

ACTIONS = {"self_harm": "support", "personal_info": "redact"}

COMPILED_IN = {k: [re.compile(p, F) for p in v] for k, v in INPUT_RULES.items()}
COMPILED_OUT = {k: [re.compile(p, F) for p in v] for k, v in OUTPUT_RULES.items()}
COMPILED_PII = [re.compile(p, fl) for p, fl in PII_PATTERNS]


@dataclass
class GuardResult:
    category: Optional[str] = None       # None = passed
    action: str = "allow"                # allow | support | redact | block
    matches: List[str] = field(default_factory=list)
    text: str = ""                       # the text to pass on (redacted when action is redact)

    @property
    def passed(self):
        return self.action in ("allow", "redact")


def _first_match(text, compiled):
    for cat, pats in compiled.items():
        hits = [m.group(0) for p in pats for m in [p.search(text)] if m]
        if hits:
            return cat, hits
    return None, []


def redact(text):
    hits = []
    for p in COMPILED_PII:
        for m in p.finditer(text):
            hits.append(m.group(0))
        text = p.sub("[private]", text)
    return text, hits


def check_input(text: str) -> GuardResult:
    cat, hits = _first_match(text, COMPILED_IN)
    if cat:
        return GuardResult(cat, ACTIONS.get(cat, "block"), hits, text)
    clean, pii = redact(text)
    if pii:
        return GuardResult("personal_info", "redact", pii, clean)
    return GuardResult(text=text)


def check_output(text: str) -> GuardResult:
    cat, hits = _first_match(text, COMPILED_OUT)
    if cat:
        return GuardResult(cat, "block", hits, text)
    return GuardResult(text=text)


def too_hard(text: str, age: str, band: dict, margin: float, min_words: int):
    """True when the answer's FKGL is far above what this age group's references use. Short answers are skipped
    because FKGL is unreliable on them. Returns (flag, fkgl or None)."""
    import textstat
    if len(text.split()) < min_words or age not in band:
        return False, None
    g = float(textstat.flesch_kincaid_grade(text))
    return g > band[age][1] + margin, g


YOUNG = {"0-3", "4-6"}
SAFE_REPLIES = {
    "self_harm": ("It sounds like you feel really sad right now. You matter, and you don't have to feel this way "
                  "alone. Please tell a grown-up you trust, like your mom, dad or teacher, right now.",
                  "I'm really sorry you're feeling this way. You matter, and you don't have to handle this alone. "
                  "Please talk to a parent, teacher or another adult you trust today. If you might hurt yourself, "
                  "call or text {crisis_line} to talk to someone right now."),
    "personal_info": ("Thanks for telling me! But things like where you live or phone numbers are private. "
                      "Let's keep those just for your family.",
                      "Quick tip: details like your address, phone number or passwords are private, so it's best not "
                      "to share them online, even with me. I didn't keep them."),
    "sexual": ("That's a question to ask your mom, dad or another grown-up you trust. Want to talk about something else?",
               "That's a good question to ask a parent, school nurse or another adult you trust. "
               "I can help with lots of other things!"),
    "violence_weapons": ("That could hurt someone, so I can't help with that. If someone is hurting you or you feel "
                         "scared, tell a grown-up you trust right away.",
                         "I can't help with that because someone could get hurt. If you're angry or scared, talking "
                         "to a parent, teacher or school counselor can really help."),
    "drugs_alcohol": ("That's not safe for kids. Let's ask a grown-up about it together!",
                      "I can't help with that, it isn't safe. If you have questions about it, a parent or school "
                      "counselor is a good person to ask."),
    "dangerous_activity": ("That could hurt you! Please ask a grown-up first before trying anything like that.",
                           "That can be really dangerous, so I can't help with it. Please check with a parent or "
                           "another adult first."),
    "contact_meeting": ("I'm a computer helper, so I can't meet anyone. Never go to meet someone from the internet "
                        "without your mom or dad!",
                        "I'm an AI, so I can't meet up or share photos. Never meet someone you only know online "
                        "without a parent knowing."),
    "prompt_injection": ("I'm here to be your friendly helper! What would you like to talk about?",
                         "I'll keep being your helpful assistant. What would you like to know?"),
    "asks_personal_info": ("Let's talk about something fun instead! What do you want to know?",
                           "Let's get back to your question. What would you like to know?"),
    "secrecy": ("You can always tell your mom, dad or another grown-up about anything we talk about!",
                "You can always share anything we talk about with your parents or another adult you trust."),
}


def safe_reply(category: str, age: str, crisis_line: str = "988") -> str:
    young, older = SAFE_REPLIES.get(category, SAFE_REPLIES["prompt_injection"])
    return (young if age in YOUNG else older).format(crisis_line=crisis_line)
