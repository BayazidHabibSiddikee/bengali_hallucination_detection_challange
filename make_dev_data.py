#!/usr/bin/env python3
"""Creates ./dev_data with the same schema as the real competition files so the
notebook can be smoke-tested locally before burning Kaggle GPU time.

Mimics real-data quirks documented in the official starter notebook:
- context stand-ins: "", "nan", "[NULL]", missing
- some responses are bare ints in the JSON
"""
import json
import random

random.seed(42)

rows = [
    # context grounding — faithful
    dict(prompt_bn="তথ্যসূত্র অনুযায়ী বাংলাদেশ কবে স্বাধীন হয়?",
         response_bn="১৯৭১ সালে বাংলাদেশ স্বাধীন হয়।",
         context="বাংলাদেশ ১৯৭১ সালের মুক্তিযুদ্ধের মাধ্যমে স্বাধীনতা অর্জন করে। ৯ মাসের যুদ্ধ শেষে ১৬ ডিসেম্বর বিজয় আসে।",
         label=1),
    # context grounding — hallucinated (contradicts context)
    dict(prompt_bn="তথ্যসূত্র অনুযায়ী বাংলাদেশ কবে স্বাধীন হয়?",
         response_bn="১৯৪৭ সালে বাংলাদেশ স্বাধীন হয়।",
         context="বাংলাদেশ ১৯৭১ সালের মুক্তিযুদ্ধের মাধ্যমে স্বাধীনতা অর্জন করে।",
         label=0),
    # factual QA — faithful
    dict(prompt_bn="বাংলাদেশের রাজধানীর নাম কী?",
         response_bn="বাংলাদেশের রাজধানী ঢাকা।",
         context="[NULL]", label=1),
    # factual QA — hallucinated
    dict(prompt_bn="বাংলাদেশের রাজধানীর নাম কী?",
         response_bn="বাংলাদেশের রাজধানী চট্টগ্রাম।",
         context="[NULL]", label=0),
    # math — faithful
    dict(prompt_bn="২ + ২ কত?", response_bn="২ + ২ = ৪", context="nan", label=1),
    # math — hallucinated
    dict(prompt_bn="২ + ২ কত?", response_bn="২ + ২ = ৫", context="", label=0),
    # translation — faithful
    dict(prompt_bn="ইংরেজিতে অনুবাদ করুন: আমি ভাত খাই।",
         response_bn="I eat rice.", context=None, label=1),
    # translation — hallucinated
    dict(prompt_bn="ইংরেজিতে অনুবাদ করুন: আমি ভাত খাই।",
         response_bn="I drive a car every morning.", context=None, label=0),
    # numeric-only response quirk (int in JSON)
    dict(prompt_bn="৩ থেকে ৫ বিয়োগ করলে কত হয়?", response_bn=-2, context="[NULL]", label=1),
    dict(prompt_bn="সাত আর তিন যোগ করলে কত?", response_bn=12, context="", label=0),
    # C1 cultural — faithful
    dict(prompt_bn="জাতীয় স্মৃতিসৌধ কোথায় অবস্থিত?",
         response_bn="জাতীয় স্মৃতিসৌধ সাভারে অবস্থিত।", context="[NULL]", label=1),
    # C1 cultural — hallucinated
    dict(prompt_bn="জাতীয় স্মৃতিসৌধ কোথায় অবস্থিত?",
         response_bn="জাতীয় স্মৃতিসৌধ সিলেটে অবস্থিত।", context="[NULL]", label=0),
    # long context, faithful summary
    dict(prompt_bn="তথ্যসূত্রটি এক বাক্যে সারাংশ করুন।",
         response_bn="ভাষা আন্দোলনে ১৯৫২ সালের ২১ ফেব্রুয়ারি ছাত্ররা প্রাণ দেন।",
         context="১৯৫২ সালের ২১ ফেব্রুয়ারি রাষ্ট্রভাষা বাংলার দাবিতে ঢাকা বিশ্ববিদ্যালয়ের ছাত্ররা মিছিল করে। "
                 "পুলিশের গুলিতে সালাম, বরকত, রফিক, জব্বারসহ অনেকে শহিদ হন। এই আন্দোলনের ফলেই বাংলা "
                 "রাষ্ট্রভাষার মর্যাদা পায় এবং পরবর্তীতে একুশে ফেব্রুয়ারি আন্তর্জাতিক মাতৃভাষা দিবস হিসেবে স্বীকৃতি পায়।",
         label=1),
    # long context, hallucinated detail
    dict(prompt_bn="তথ্যসূত্রটি এক বাক্যে সারাংশ করুন।",
         response_bn="১৯৫২ সালের ভাষা আন্দোলন হয়েছিল কলকাতায় এবং কেউ হতাহত হয়নি।",
         context="১৯৫২ সালের ২১ ফেব্রুয়ারি রাষ্ট্রভাষা বাংলার দাবিতে ঢাকা বিশ্ববিদ্যালয়ের ছাত্ররা মিছিল করে। "
                 "পুলিশের গুলিতে সালাম, বরকত, রফিকসহ অনেকে শহিদ হন।",
         label=0),
    dict(prompt_bn="সূর্য কোন দিকে ওঠে?", response_bn="সূর্য পূর্ব দিকে ওঠে।", context="[NULL]", label=1),
    dict(prompt_bn="সূর্য কোন দিকে ওঠে?", response_bn="সূর্য পশ্চিম দিকে ওঠে।", context="[NULL]", label=0),
    dict(prompt_bn="পানির রাসায়নিক সংকেত কী?", response_bn="পানির সংকেত H2O।", context="", label=1),
    dict(prompt_bn="পানির রাসায়নিক সংকেত কী?", response_bn="পানির সংকেত CO2।", context="", label=0),
    dict(prompt_bn="একুশে ফেব্রুয়ারি কী দিবস?",
         response_bn="একুশে ফেব্রুয়ারি আন্তর্জাতিক মাতৃভাষা দিবস।", context="[NULL]", label=1),
    dict(prompt_bn="একুশে ফেব্রুয়ারি কী দিবস?",
         response_bn="একুশে ফেব্রুয়ারি বিশ্ব স্বাস্থ্য দিবস।", context="[NULL]", label=0),
    dict(prompt_bn="তথ্যসূত্র অনুযায়ী পদ্মা সেতুর দৈর্ঘ্য কত?",
         response_bn="পদ্মা সেতুর দৈর্ঘ্য ৬.১৫ কিলোমিটার।",
         context="পদ্মা সেতু বাংলাদেশের দীর্ঘতম সেতু, যার দৈর্ঘ্য ৬.১৫ কিলোমিটার। এটি ২০২২ সালে উদ্বোধন করা হয়।",
         label=1),
    dict(prompt_bn="তথ্যসূত্র অনুযায়ী পদ্মা সেতুর দৈর্ঘ্য কত?",
         response_bn="পদ্মা সেতুর দৈর্ঘ্য ১২ কিলোমিটার।",
         context="পদ্মা সেতু বাংলাদেশের দীর্ঘতম সেতু, যার দৈর্ঘ্য ৬.১৫ কিলোমিটার।",
         label=0),
    dict(prompt_bn="১০ এর অর্ধেক কত?", response_bn="১০ এর অর্ধেক ৫।", context="", label=1),
    dict(prompt_bn="১০ এর অর্ধেক কত?", response_bn="১০ এর অর্ধেক ৭।", context="", label=0),
    dict(prompt_bn="বাংলা অনুবাদ করুন: Good morning.", response_bn="শুভ সকাল।", context=None, label=1),
    dict(prompt_bn="বাংলা অনুবাদ করুন: Good morning.", response_bn="শুভ রাত্রি, বন্ধুরা।", context=None, label=0),
    dict(prompt_bn="জাতীয় ফুল কী?", response_bn="বাংলাদেশের জাতীয় ফুল শাপলা।", context="[NULL]", label=1),
    dict(prompt_bn="জাতীয় ফুল কী?", response_bn="বাংলাদেশের জাতীয় ফুল গোলাপ।", context="[NULL]", label=0),
    dict(prompt_bn="তথ্যসূত্র অনুযায়ী মুজিবনগর সরকার কবে গঠিত হয়?",
         response_bn="১৯৭১ সালের ১০ এপ্রিল মুজিবনগর সরকার গঠিত হয়।",
         context="১৯৭১ সালের ১০ এপ্রিল প্রবাসী বাংলাদেশ সরকার (মুজিবনগর সরকার) গঠিত হয় এবং ১৭ এপ্রিল শপথ গ্রহণ করে।",
         label=1),
    dict(prompt_bn="তথ্যসূত্র অনুযায়ী মুজিবনগর সরকার কবে গঠিত হয়?",
         response_bn="মুজিবনগর সরকার গঠিত হয় ১৯৭৫ সালের ১৫ আগস্ট ঢাকায়।",
         context="১৯৭১ সালের ১০ এপ্রিল প্রবাসী বাংলাদেশ সরকার (মুজিবনগর সরকার) গঠিত হয়।",
         label=0),
]

# duplicate the pool so the ML meta-classifier path (needs >=40 rows) is exercised locally
rows = rows + [dict(r) for r in rows]
BD_MARKERS = ("বাংলাদেশ", "স্মৃতিসৌধ", "মুজিবনগর", "পদ্মা", "ভাষা", "শাপলা", "একুশে")
for i, r in enumerate(rows):
    r["id"] = i
    # band metadata exists in the labeled sample but not the test csv -> exercises
    # the notebook's "train-only feature must be excluded" guard
    r["cultural_band"] = "C1" if any(m in str(r["prompt_bn"]) for m in BD_MARKERS) else "C0"

random.shuffle(rows)

import os
os.makedirs("dev_data", exist_ok=True)
with open("dev_data/dataset samples.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

# test split: reuse a few rows verbatim (leak-check path) + label-free variants
import csv
test_rows = []
for i, r in enumerate(rows[:12]):
    test_rows.append(dict(id=1000 + i, prompt_bn=str(r["prompt_bn"]),
                          response_bn=str(r["response_bn"]),
                          context="" if r["context"] in (None, "nan", "[NULL]") else r["context"]))
test_rows.append(dict(id=2000, prompt_bn="চাঁদে প্রথম মানুষ কে?",
                      response_bn="নীল আর্মস্ট্রং চাঁদে প্রথম পা রাখেন।", context=""))
test_rows.append(dict(id=2001, prompt_bn="চাঁদে প্রথম মানুষ কে?",
                      response_bn="ইউরি গ্যাগারিন চাঁদে প্রথম পা রাখেন।", context="[NULL]"))

with open("dev_data/test set.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "prompt_bn", "response_bn", "context"])
    w.writeheader()
    w.writerows(test_rows)

with open("dev_data/sample submission.csv", "w", encoding="utf-8", newline="") as f:
    f.write("id,label\n")
    for r in test_rows:
        f.write(f"{r['id']},1\n")

print(f"dev_data written: {len(rows)} train rows, {len(test_rows)} test rows")
