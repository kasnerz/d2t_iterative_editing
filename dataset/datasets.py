#!/usr/bin/env python3

import json
import csv
import random

import os
import pathlib
from collections import defaultdict, namedtuple
from sentence_scorer import SentenceScorer
import webnlg_parsing

# wrappers for datasets

class Dataset:
    def __init__(self, path, fold, sort_by_lens, name):
        self.name = name
        self.path = path
        self.fold = fold
        self.sort_by_lens = sort_by_lens
        self.data = None
        self.lens = None
        self.idxs = None
        self.templates = None
        self.templates_filename = os.path.join(self.path,
                                    "templates.json")
        self.default_templates = [
            "The <predicate> of <subject> is <object> ."
        ]

        os.makedirs(path, exist_ok=True)

        self.load()


    def load_templates(self):
        try:
            with open(self.templates_filename) as json_file:
                self.templates = json.load(json_file)
        except FileNotFoundError:
            return self.extract_templates()

    def load(self):
        raise NotImplementedError

    def get_templates(self, triple):
        raise NotImplementedError

    def extract_templates(self):
        raise NotImplementedError

    def sort_by_lens(self):
        self.data.sort(key=lambda el: len(el['triples']))

        # find beginning of n-triples in the list for each n
        lens_list = [len(el['triples']) for el in self.data]
        self.lens = sorted(list(set(lens_list)))

        print(f"Available lengths: {self.lens}")

        self.idxs = [lens_list.index(x) for x in self.lens]

        print(f"Indexes: {self.idxs}")

        print(f"Loaded {len(self.data)} items.")



class WebNLGDataset(Dataset):
    def __init__(self, path, fold, sort_by_lens=True):
        super().__init__(path, fold, sort_by_lens, name="webnlg")

    def extract_templates(self):
        if self.fold != "train":
            raise NotImplementedError(f"Will not extract templates from {self.fold} dataset")

        l = []
        templates = defaultdict(set)

        for entry in self.data:
            n = len(entry["triples"])

            if n > 1:
                break

            triple = entry["triples"][0]
            subj, pred, obj = triple


            for lex in entry["lex_list"]:
                ner2ent_items = list(lex["ner2ent"].items())

                if len(ner2ent_items) != 2:
                    # print(f"Warning: corrupted entry {lex}")
                    continue

                if ner2ent_items[0][1] == subj and ner2ent_items[1][1] == obj:
                    subj_ent = ner2ent_items[0][0]
                    obj_ent = ner2ent_items[1][0]
                elif ner2ent_items[0][1] == obj and ner2ent_items[1][1] == subj:
                    obj_ent = ner2ent_items[0][0]
                    subj_ent = ner2ent_items[1][0]
                else:
                    continue

                template = lex["target"].replace(subj_ent, "<subject>") \
                                          .replace(obj_ent, "<object>")
                if template:
                    templates[pred].add(template)

        templates = {key: list(value) for key, value in templates.items()}

        with open(self.templates_filename, "w") as json_file:
            json.dump(templates,json_file,
                    indent=4, separators=(',', ': '), sort_keys=True)

        return templates


    def load(self):
        self.data = []
        data_dir = os.path.join(self.path, "src", f"{self.fold}")
        xml_entryset = webnlg_parsing.run_parser(data_dir)

        for xml_entry in xml_entryset:
            triples = [[e.subject, e.predicate, e.object]
                for e in xml_entry.modifiedtripleset]
            lex_list = []

            for lex in xml_entry.lexEntries:
                target_txt = lex.text
                target = lex.template

                ner2ent = {
                    ref.tag : ref.entity for ref in lex.references
                }

                lex = {
                    "target_txt" : target_txt,
                    "target" : target,
                    "ner2ent" : ner2ent
                }
                lex_list.append(lex)

            if not any([lex['target_txt'] for lex in lex_list]):
                print(f"Warning: Entry does not contain any lexicalizations, skipping.\n{triples}")
                continue

            entry = {
                "triples" : triples,
                "lex_list" : lex_list
            }
            self.data.append(entry)

        if self.sort_by_lens:
            Dataset.sort_by_lens(self)

    def get_templates(self, triple):
        pred = triple[1]

        if pred in self.templates:
            templates = self.templates[pred]
        else:
            templates = self.default_templates

        return templates



class E2EDataset(Dataset):
    def __init__(self, path, fold, sort_by_lens=True):
        super().__init__(path, fold, sort_by_lens, name="e2e")
        self.templates_basic_filename = os.path.join(self.path,
                                    "templates_basic.json")

        self.sentence_scorer = SentenceScorer(device="cpu")

    def load_templates(self):
        try:
            with open(self.templates_basic_filename) as json_file:
                self.templates_basic = json.load(json_file)
        except FileNotFoundError:
            print(f"Cannot find basic templates for E2E dataset: {self.templates_basic_filename}")

        try:
            with open(self.templates_filename) as json_file:
                self.templates = json.load(json_file)
        except FileNotFoundError:
            return self.extract_templates()


    def _mr_to_triples(self, mr):
        triples = []

        # cannot be dictionary, slot keys can be duplicated
        items = [x.strip() for x in mr.split(",")]
        subj = None

        keys = []
        vals = []

        for item in items:
            key, val = item.split("[")
            val = val[:-1]

            keys.append(key)
            vals.append(val)

        name_idx = None if "name" not in keys else keys.index("name")
        eatType_idx = None if "eatType" not in keys else keys.index("eatType")

        if name_idx is not None:
            subj = vals[name_idx]
            del keys[name_idx]
            del vals[name_idx]

            # yet another corrupted case hotfix
            if not keys:
                keys.append("eatType")
                vals.append("restaurant")

        elif eatType_idx is not None:
            subj = vals[eatType_idx]
            del keys[eatType_idx]
            del vals[eatType_idx]
        else:
            print("Warning: cannot recognize subject in mr", mr)
            # hotfix for corrupted pairs
            subj = "Restaurant"

        for key, val in zip(keys, vals):
            triples.append((subj, key, val))

        return tuple(sorted(triples))


    def load(self):
        triples_to_lex = defaultdict(list)

        with open(os.path.join(self.path, "src", f"{self.fold}.csv")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')

            # skip header
            next(csv_reader)

            self.data = []

            err = 0

            for i, line in enumerate(csv_reader):
                triples = self._mr_to_triples(line[0])

                if not triples or len(triples) == 1:
                    err += 1

                    # cannot skip for dev and test
                    if self.fold == "train":
                        continue

                lex = {
                    "target_txt" : line[1]
                }
                triples_to_lex[triples].append(lex)

            for triples, lex_list in triples_to_lex.items():
                entry = {
                    "triples" : triples,
                    "lex_list" : lex_list
                }
                self.data.append(entry)

        print(f"Warning: {err} corrupted instances")

        if self.sort_by_lens:
            Dataset.sort_by_lens(self)


    def select_triples_for_double_template(self, triples):
        sorted_triples = sorted(triples, key=lambda t:t[1])
        triple_pairs = [(s,t,i,j) for j,t in enumerate(sorted_triples) for i,s in enumerate(sorted_triples) if s[1] < t[1]]

        templates = list(filter(lambda x: x[0] is not None,
            [(self.get_templates_multiple(tp[:2]),tp[2],tp[3]) for tp in triple_pairs]))

        if templates:
            return random.choice(templates)

        return None



    def get_key_multiple(self, sorted_triples):
        slots = []
        for triple in sorted_triples:

            if triple[1] == "familyFriendly":
                slot = ("is" if triple[2] == "yes" else "not") + "FamilyFriendly"
            else:
                slot = triple[1]

            slots.append(slot)

        key = ", ".join(slots)

        return key


    def get_templates_multiple(self, triples):
        sorted_triples = sorted(triples, key=lambda t:t[1])

        key = self.get_key_multiple(sorted_triples)

        if key in self.templates:
            templates = self.templates[key]

            # TODO find a better way for speeding it up
            max_templates = 50
            if len(templates) > max_templates:
                templates = random.sample(templates, max_templates)

            return templates

        return None



    def get_templates(self, triple):
        pred = triple[1]
        val = triple[2]

        if pred in self.templates_basic:
            templates = self.templates_basic[pred]

            if type(templates) is dict and val in templates:
                templates = templates[val]
        else:
            templates = self.default_templates

        return templates


    def extract_templates(self):
        if self.fold != "train":
            raise NotImplementedError(f"Will not extract templates from {self.fold} dataset")

        l = []
        entry_list = self.data
        templates = defaultdict(set)

        for entry in entry_list:
            n = len(entry["triples"])

            if n > 2:
                break

            sorted_triples = sorted([list(triple) for triple in entry["triples"]], key=lambda t:t[1])

            # special case, bool value
            for triple in sorted_triples:
                if triple[1] == "familyFriendly":
                    triple[1] = ("is" if triple[2] == "yes" else "not") + "FamilyFriendly"

                    # should not appear in the sentence
                    triple[2] = "<plh>"

            key = self.get_key_multiple(sorted_triples)

            subj1, prop1, obj1 = sorted_triples[0]
            subj2, prop2, obj2 = sorted_triples[1]

            assert subj1 == subj2

            lexicalizations = [lex['target_txt'] for lex in entry["lex_list"]]

            for lex in lexicalizations:
                if subj1 in lex and \
                    (obj1 in lex or obj1 == "<plh>") and \
                    (obj2 in lex or obj2 == "<plh>"):
                    template = lex.replace(subj1, "<subject>") \
                                          .replace(obj1, "<object1>") \
                                          .replace(obj2, "<object2>")
                    templates[key].add(template)

        templates = {key: list(value) for key, value in templates.items()}

        with open(self.templates_filename, "w") as json_file:
            json.dump(templates,json_file,
                    indent=4, separators=(',', ': '), sort_keys=True)

        return templates
