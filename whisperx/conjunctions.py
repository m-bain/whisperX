# conjunctions.py

conjunctions_by_language = {
    'en': {'and', 'whether', 'or', 'as', 'but', 'so', 'for', 'nor', 'which', 'yet', 'although', 'since', 'unless', 'when', 'while', 'because', 'if', 'how', 'that', 'than', 'who', 'where', 'what', 'near', 'before', 'after', 'across', 'through', 'until', 'once', 'whereas', 'even', 'both', 'either', 'neither', 'though'},
    'fr': {'et', 'ou', 'mais', 'parce', 'bien', 'pendant', 'quand', 'où', 'comme', 'si', 'que', 'avant', 'après', 'aussitôt', 'jusqu’à', 'à', 'malgré', 'donc', 'tant', 'puisque', 'ni', 'soit', 'bien', 'encore', 'dès', 'lorsque'},
    'de': {'und', 'oder', 'aber', 'weil', 'obwohl', 'während', 'wenn', 'wo', 'wie', 'dass', 'bevor', 'nachdem', 'sobald', 'bis', 'außer', 'trotzdem', 'also', 'sowie', 'indem', 'weder', 'sowohl', 'zwar', 'jedoch'},
    'es': {'y', 'o', 'pero', 'porque', 'aunque', 'sin', 'mientras', 'cuando', 'donde', 'como', 'si', 'que', 'antes', 'después', 'tan', 'hasta', 'a', 'a', 'por', 'ya', 'ni', 'sino'},
    'it': {'e', 'o', 'ma', 'perché', 'anche', 'mentre', 'quando', 'dove', 'come', 'se', 'che', 'prima', 'dopo', 'appena', 'fino', 'a', 'nonostante', 'quindi', 'poiché', 'né', 'ossia', 'cioè'},
    'ja': {'そして', 'または', 'しかし', 'なぜなら', 'もし', 'それとも', 'だから', 'それに', 'なのに', 'そのため', 'かつ', 'それゆえに', 'ならば', 'もしくは', 'ため'},
    'zh': {'和', '或', '但是', '因为', '任何', '也', '虽然', '而且', '所以', '如果', '除非', '尽管', '既然', '即使', '只要', '直到', '然后', '因此', '不但', '而是', '不过'},
    'nl': {'en', 'of', 'maar', 'omdat', 'hoewel', 'terwijl', 'wanneer', 'waar', 'zoals', 'als', 'dat', 'voordat', 'nadat', 'zodra', 'totdat', 'tenzij', 'ondanks', 'dus', 'zowel', 'noch', 'echter', 'toch'},
    'uk': {'та', 'або', 'але', 'тому', 'хоча', 'поки', 'бо', 'коли', 'де', 'як', 'якщо', 'що', 'перш', 'після', 'доки', 'незважаючи', 'тому', 'ані'},
    'pt': {'e', 'ou', 'mas', 'porque', 'embora', 'enquanto', 'quando', 'onde', 'como', 'se', 'que', 'antes', 'depois', 'assim', 'até', 'a', 'apesar', 'portanto', 'já', 'pois', 'nem', 'senão'},
    'ar': {'و', 'أو', 'لكن', 'لأن', 'مع', 'بينما', 'عندما', 'حيث', 'كما', 'إذا', 'الذي', 'قبل', 'بعد', 'فور', 'حتى', 'إلا', 'رغم', 'لذلك', 'بما'},
    'cs': {'a', 'nebo', 'ale', 'protože', 'ačkoli', 'zatímco', 'když', 'kde', 'jako', 'pokud', 'že', 'než', 'poté', 'jakmile', 'dokud', 'pokud ne', 'navzdory', 'tak', 'stejně', 'ani', 'tudíž'},
    'ru': {'и', 'или', 'но', 'потому', 'хотя', 'пока', 'когда', 'где', 'как', 'если', 'что', 'перед', 'после', 'несмотря', 'таким', 'также', 'ни', 'зато'},
    'pl': {'i', 'lub', 'ale', 'ponieważ', 'chociaż', 'podczas', 'kiedy', 'gdzie', 'jak', 'jeśli', 'że', 'zanim', 'po', 'jak tylko', 'dopóki', 'chyba', 'pomimo', 'więc', 'tak', 'ani', 'czyli'},
    'hu': {'és', 'vagy', 'de', 'mert', 'habár', 'míg', 'amikor', 'ahol', 'ahogy', 'ha', 'hogy', 'mielőtt', 'miután', 'amint', 'amíg', 'hacsak', 'ellenére', 'tehát', 'úgy', 'sem', 'vagyis'},
    'fi': {'ja', 'tai', 'mutta', 'koska', 'vaikka', 'kun', 'missä', 'kuten', 'jos', 'että', 'ennen', 'sen jälkeen', 'heti', 'kunnes', 'ellei', 'huolimatta', 'siis', 'sekä', 'eikä', 'vaan'},
    'fa': {'و', 'یا', 'اما', 'چون', 'اگرچه', 'در حالی', 'وقتی', 'کجا', 'چگونه', 'اگر', 'که', 'قبل', 'پس', 'به محض', 'تا زمانی', 'مگر', 'با وجود', 'پس', 'همچنین', 'نه'},
    'el': {'και', 'ή', 'αλλά', 'επειδή', 'αν', 'ενώ', 'όταν', 'όπου', 'όπως', 'αν', 'που', 'προτού', 'αφού', 'μόλις', 'μέχρι', 'εκτός', 'παρά', 'έτσι', 'όπως', 'ούτε', 'δηλαδή'},
    'tr': {'ve', 'veya', 'ama', 'çünkü', 'her ne', 'iken', 'nerede', 'nasıl', 'eğer', 'ki', 'önce', 'sonra', 'hemen', 'kadar', 'rağmen', 'hem', 'ne', 'yani'},
    'da': {'og', 'eller', 'men', 'fordi', 'selvom', 'mens', 'når', 'hvor', 'som', 'hvis', 'at', 'før', 'efter', 'indtil', 'medmindre', 'således', 'ligesom', 'hverken', 'altså'},
    'he': {'ו', 'או', 'אבל', 'כי', 'אף', 'בזמן', 'כאשר', 'היכן', 'כיצד', 'אם', 'ש', 'לפני', 'אחרי', 'ברגע', 'עד', 'אלא', 'למרות', 'לכן', 'כמו', 'לא', 'אז'},
    'vi': {'và', 'hoặc', 'nhưng', 'bởi', 'mặc', 'trong', 'khi', 'ở', 'như', 'nếu', 'rằng', 'trước', 'sau', 'ngay', 'cho', 'trừ', 'mặc', 'vì', 'giống', 'cũng', 'tức'},
    'ko': {'그리고', '또는','그런데','그래도', '이나', '결국', '마지막으로', '마찬가지로', '반면에', '아니면', '거나', '또는', '그럼에도', '그렇기', '때문에', '덧붙이자면', '게다가', '그러나',  '고', '그래서', '랑', '한다면', '하지만', '무엇', '왜냐하면', '비록', '동안', '언제', '어디서', '어떻게', '만약', '그', '전에', '후에', '즉시', '까지', '아니라면', '불구하고', '따라서', '같은', '도'},
    'ur': {'اور', 'یا', 'مگر', 'کیونکہ', 'اگرچہ', 'جبکہ', 'جب', 'کہاں', 'کس طرح', 'اگر', 'کہ', 'سے پہلے', 'کے بعد', 'جیسے ہی', 'تک', 'اگر نہیں تو', 'کے باوجود', 'اس لئے', 'جیسے', 'نہ'},
    'hi': {'और', 'या', 'पर', 'तो', 'न', 'फिर', 'हालांकि', 'चूंकि', 'अगर', 'कैसे', 'वह', 'से', 'जो', 'जहां', 'क्या', 'नजदीक', 'पहले', 'बाद', 'के', 'पार', 'माध्यम', 'तक', 'एक', 'जबकि', 'यहां', 'तक', 'दोनों', 'या', 'न', 'हालांकि'}

}

commas_by_language = {
    'ja': '、', 
    'zh': '，',
    'fa': '،', 
    'ur': '،'  
}

def get_conjunctions(lang_code):
    return conjunctions_by_language.get(lang_code, set())

def get_comma(lang_code):
    return commas_by_language.get(lang_code, ',')