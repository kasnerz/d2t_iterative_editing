#!/usr/bin/env python3
# -"- encoding: utf-8 -"-

import re
import json

regex_dict = {
    'area': {
        'city centre': [
            '(?:city|town) cent(?:re|er)',
            'cent(?:re|er) of (?:the )?(?:city|town)',
            'in the cent(?:re|er)',
        ],
        'riverside': [
            'riverside',
            '(?:near|by|at|close to|along|on|off|beside) the river',
        ],
    },
    'eatType': {
        'coffee shop': [
            'coffee[- ]+shop',
            'caf[eé]',
            'coffee',
        ],
        'pub': [
            'pub',
        ],
        'restaurant': [
            'restaurant',
        ],
    },
    'familyFriendly': {
        'no': [
            '(?:isn\'t|not|non|no)[ -]+(?:\w+ ){0,2}(?:child|children|family|kids|kid)[ -]+(?:friendly|orien(?:ta)?ted)',
            '(?:child|children|family|kids|kid)[ -]+unfriendly',
            'adults?[ -]+only',
            'only for adults',
            '(?:no|not) (?:kids|children|famil(?:y|ies))',
            '(?:not|no)(?: good| suitable| friendly| orien(?:ta)?ted| open(?:ed))? (?:at|for|to|with)(?: the)? (?:kids|children|family|families|all age)',
            '(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is)(?:n\'t| not) (?:welcome|allowed|accepted)',
            '(?:does not|doesn\'t) (?:welcome|allow|accept) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)',
            'adult (?:establishment|venue|place|establish)',
        ],
        'yes': [
            'for (?:kids|children|family|families)',
            'family place',
            'place to bring the(?: whole)? family',
            '(?:friendly|suitable|good|orien(?:ta)?ted|open(?:ed)) (?:at|with|to|for)(?: the)(?:kids?|child(?:ren)?|famil(?:y|ies)?|all age)',
            '(?:child|children|family|kids|kid)[ -]+(?:friendly|orien(?:ta)?ted)',
            '(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is) (?:welcome|allowed|accepted)',
            '(?:welcomes?|allows?|accepts?) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)',
        ],
    },
    'food': {
        'Chinese': ['Chinese', 'Chines'],
        'English': ['English', 'British'],
        'Fast food': ['Fast food'],
        'French': ['French'],
        'Indian': ['Indian'],
        'Italian': ['Italian'],
        'Japanese': ['Japanese'],
    },
    'name': [
        'Alimentum',
        'Aromi',
        'Bibimbap House',
        'Blue Spice',
        'Browns Cambridge',
        'Clowns',
        'Cocum',
        'Cotto',
        'Fitzbillies',
        'Giraffe',
        'Green Man',
        'Loch Fyne',
        'Midsummer House',
        'Strada',
        'Taste of Cambridge',
        'The Cambridge Blue',
        'The Cricketers',
        'The Dumpling Tree',
        'The Eagle',
        'The Golden Curry',
        'The Golden Palace',
        'The Mill',
        'The Olive Grove',
        'The Phoenix',
        'The Plough',
        'The Punter',
        'The Rice Boat',
        'The Twenty Two',
        'The Vaults',
        'The Waterman',
        'The Wrestlers',
        'Travellers Rest Beefeater',
        'Wildwood',
        'Zizzi',
    ],
    'near': {
        "near": [
            'near',
            'close',
        ]
    },
    'priceRange': {
        "cheap": [
            "(?:inexpensive|cheap)(?:ly)?",
            "low[- ]+price[ds]?",
            "affordabl[ey]",
            "prices?(?: range)?(?: \w+){0,3} low",
        ],
        "less than £20": [
            "(?:inexpensive|cheap)(?:ly)?",
            "affordabl[ey]",
            "(?:less than|under) £? *20",
            "moderately priced",
            "low[- ]+price[ds]?",
            "prices?(?: range)?(?: \w+){0,3} low",
        ],
        "more than £30": [
            "(?:more than|over) £? *30",
            "high[- ]+price[ds]?",
            "expensive",
            "not cheap",
            "prices?(?: range)?(?: \w+){0,3} high",
        ],
        "high": [
            "high[- ]+price[ds]?",
            "expensive",
            "prices?(?: range)?(?: \w+){0,3} high",
        ],
        "moderate": [
            "(?:moderate|reasonable|ok|average)(?:ly)?[- ]+price[ds]?",
            "not cheap",
            "affordable",
            "mid[- ]+(?:range[- ]+)price[ds]?",
            "prices?(?: range)?(?: \w+){0,3} (?:ok|average|moderate|reasonable)",
        ],
        "£20-25": [
            "£? *20 *(?:[-–]*|to) *25",
            "(?:moderate|reasonable|ok|average)(?:ly)?[- ]+price[ds]?",
            "prices?(?: range)?(?: \w+){0,3} (?:ok|average|moderate|reasonable)",
            "affordable",
        ]
    },
    'customer rating': {
        "1 out of 5": [
            "(?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            "(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?(?:low|bad|poor)(?:(?: \w+){0,3} (?:1|one)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            "(?:low|bad|poor|(?:not|doesn't|isn't)(?: \w+){0,2} (:?good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "3 out of 5": [
            "(?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            "(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?average(?:(?: \w+){0,3} (?:3|three)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            "(?:average|(?<!very )(?:good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "5 out of 5": [
            "(?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            "(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?high(?:(?: \w+){0,3} (?:5|five)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            "(?:high|excellent|very good|great)(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "high": [
            "(?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            "(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?high",
            "(?:high|excellent|very good|great|well)(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
        "average": [
            "(?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            "(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?average",
            "(?:average|(?<!very )(?:good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
        "low": [
            "(?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            "(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?(?:low|bad|poor)",
            "(?:low|bad|poor|(?:not|doesn't|isn't)(?: \w+){0,2} (?:well|good))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
    },
}


def compile_patterns(patterns):
    """Compile a list of patterns into one big option regex. Note that all of them will match whole words only."""
    # pad intent patterns with \b (word boundary), unless they contain '^'/'$' (start/end)
    return re.compile('|'.join([((r'\b' if not pat.startswith('^') else '') + pat +
                                 (r'\b' if not pat.endswith('$') else ''))
                                for pat in patterns]),
                      re.I | re.UNICODE)

def check_facts(sent, triples):
    subj = triples[0].subj

    if not subj in sent:
        return False

    for triple in triples:
        # corrputed example, should not happen
        if triple.pred == "name":
            continue

        if triple.pred == "near":
            if triple.obj not in sent:
                return False
            triple.obj = "near"

        assert triple.pred in regex_dict, f"Cannot find {triple.pred} in {regex_dict.keys()}"
        assert triple.obj in regex_dict[triple.pred], f"Cannot find {triple.obj} in {regex_dict[triple.pred]}"

        pattern_list = regex_dict[triple.pred][triple.obj]
        pattern = compile_patterns(pattern_list)

        res = re.search(pattern, sent)

        if not res:
            return False

    return True


def get_missing(sent, triples):
    subj = triples[0].subj

    missing = []

    for triple in triples:
        # corrputed example, should not happen
        if triple.pred == "name":
            continue

        if triple.pred == "near":
            if triple.obj not in sent:
                missing.append(triple)
            continue

        assert triple.pred in regex_dict, f"Cannot find {triple.pred} in {regex_dict.keys()}"
        assert triple.obj in regex_dict[triple.pred], f"Cannot find {triple.obj} in {regex_dict[triple.pred]}"

        pattern_list = regex_dict[triple.pred][triple.obj]
        pattern = compile_patterns(pattern_list)

        res = re.search(pattern, sent)

        if not res:
            missing.append(triple)

    return missing