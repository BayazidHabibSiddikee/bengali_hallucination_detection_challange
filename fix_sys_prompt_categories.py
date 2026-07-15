"""
Update build_sys_prompt() to add specific Bengali history prompt
and improve base_rule with better few-shot examples per category.
"""
import json

NEW_BUILD_SYS = '''    def build_sys_prompt(category):
        examples = {
            "comprehension": (
                "প্রশ্ন: মেহদী হাসান খান কোন বিশ্ববিদ্যালয়ের ছাত্র?\\n"
                "অনুচ্ছেদ: মেহদী হাসান খান ময়মনসিংহ মেডিকেল কলেজের একজন ছাত্র।\\n"
                "উত্তর: ময়মনসিংহ মেডিকেল কলেজ → Verdict: 1\\n"
                "উত্তর: ঢাকা বিশ্ববিদ্যালয় → Verdict: 0"
            ),
            "vocabulary": (
                "প্রশ্ন: 'কাঁচা সোনা' বাগধারার অর্থ কী?\\n"
                "উত্তর: অপরিশোধিত স্বর্ণ → Verdict: 1\\n"
                "উত্তর: তাজা শাকসবজি → Verdict: 0"
            ),
            "history": (
                "প্রশ্ন: বাংলাদেশের মুক্তিযুদ্ধ কত সালে হয়?\\n"
                "উত্তর: ১৯৭১ সালে → Verdict: 1\\n"
                "উত্তর: ১৯৪৭ সালে → Verdict: 0"
            ),
            "math": (
                "প্রশ্ন: ২৫ এর ২০% কত?\\n"
                "উত্তর: ৫ → Verdict: 1\\n"
                "উত্তর: ৫০ → Verdict: 0"
            ),
            "code_mixed": (
                "প্রশ্ন: Python-এ list এর length বের করার function কী?\\n"
                "উত্তর: len() → Verdict: 1\\n"
                "উত্তর: size() → Verdict: 0"
            ),
            "general_knowledge": (
                "প্রশ্ন: বাংলাদেশের রাজধানী কোথায়?\\n"
                "উত্তর: ঢাকা → Verdict: 1\\n"
                "উত্তর: চট্টগ্রাম → Verdict: 0"
            ),
        }
        base_rule = (
            "কোনো ব্যাখ্যা দেবেন না। শুধুমাত্র 0 অথবা 1 লিখুন।\\n"
            "উদাহরণ:\\n" + examples.get(category, examples["general_knowledge"])
        )

        if category == "code_mixed":
            return (
                "আপনি একজন বহুভাষিক হ্যালুসিনেশন বিশ্লেষক। "
                "প্রশ্ন বা উত্তরে বাংলা, ইংরেজি বা বাংলিশের মিশ্রণ থাকতে পারে। "
                "উত্তরটি সঠিক হলে শুধু '1', ভুল হলে শুধু '0' লিখুন। " + base_rule
            )

        elif category == "comprehension":
            return (
                "আপনি একটি নির্ভুল হ্যালুসিনেশন সনাক্তকরণ এআই। "
                "দেওয়া অনুচ্ছেদ থেকে প্রশ্নের উত্তরটি সঠিকভাবে যাচাই করুন। "
                "অনুচ্ছেদের তথ্যের সাথে মিলে গেলে '1', না মিললে বা অতিরিক্ত তথ্য থাকলে '0' লিখুন। " + base_rule
            )

        elif category == "vocabulary":
            return (
                "আপনি একজন বিশেষজ্ঞ বাংলা ভাষাবিদ। "
                "বাংলা শব্দ, বাগধারা, ব্যাকরণ বা ভাষাতাত্ত্বিক প্রশ্নের উত্তর যাচাই করুন। "
                "এটি C1 সাংস্কৃতিক-ভাষাগত বিভাগ। সঠিক হলে '1', ভুল হলে '0' লিখুন। " + base_rule
            )

        elif category == "history":
            return (
                "আপনি একজন বাংলাদেশ ও বাংলা ইতিহাসের বিশেষজ্ঞ। "
                "ঐতিহাসিক তথ্য, সাল, ঘটনা ও ব্যক্তিত্ব যাচাই করুন। "
                "তথ্য সঠিক হলে '1', ভুল বা বানোয়াট হলে '0' লিখুন। " + base_rule
            )

        elif category == "math":
            return (
                "আপনি একজন গাণিতিক মূল্যায়নকারী এআই। "
                "গাণিতিক হিসাব ও উত্তর নিখুঁতভাবে যাচাই করুন। "
                "সম্পূর্ণ সঠিক হলে '1', সামান্যতম ভুল থাকলে '0' লিখুন। " + base_rule
            )

        else:  # general_knowledge
            return (
                "আপনি একজন কঠোর তথ্য-যাচাইকারী। "
                "সাধারণ জ্ঞান ও বাস্তবিক তথ্যের সত্যতা যাচাই করুন। "
                "সম্পূর্ণ সত্য ও নির্ভুল হলে '1', ভুল বা মনগড়া হলে '0' লিখুন। " + base_rule
            )
'''

def fix_build_sys(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = cell.get('source', [])
        src_str = ''.join(src)
        
        if 'def build_sys_prompt' in src_str:
            new_src = []
            in_build_sys = False
            skip = False
            
            for line in src:
                if '    def build_sys_prompt(' in line:
                    in_build_sys = True
                    skip = True
                    # inject new function
                    for new_line in NEW_BUILD_SYS.split('\n'):
                        new_src.append(new_line + '\n')
                    continue
                
                if skip:
                    # Wait for the end of the old build_sys_prompt
                    if '    def one_pass()' in line:
                        skip = False
                        new_src.append(line)
                    continue
                
                new_src.append(line)
            
            nb['cells'][i]['source'] = new_src
            print(f"Fixed build_sys_prompt in {filename} cell {i}")

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix_build_sys('pipeline.ipynb')
fix_build_sys('bengali-hallu.ipynb')
