import pandas as pd
from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import pickle
import os



# ======== تحميل موارد NLTK ========
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ======== إعداد Flask ========
app = Flask(__name__)

# ======== تحميل البيانات ========
df = pd.read_json('data chat bot.json', encoding='utf-8-sig')

# ======== تحميل النموذج عند بدء التطبيق فقط ========
model = None
model_file = 'chatbot_model.pkl'  # اسم الملف لحفظ النموذج

def load_model():
    global model
    if model is None:
        # إذا كان الملف موجودًا، حمل النموذج منه
        if os.path.exists(model_file):
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
                print("تم تحميل النموذج من الملف.")
        else:
            # إذا لم يكن الملف موجودًا، حمل النموذج من الإنترنت واحفظه
            print("تحميل النموذج من الإنترنت...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            # حفظ النموذج في ملف
            with open(model_file, 'wb') as file:
                pickle.dump(model, file)
                print("تم حفظ النموذج إلى الملف.")

# ======== دالة تطبيع النص ========
def normalize_text(text):
    if not isinstance(text, str):  # إذا كانت القيمة ليست نصًا، حولها إلى نص
        text = str(text)

        # Normalize specific terms أولًا لتعطي الأولوية للمصطلحات الدقيقة
    specific_replacements = {
        r'\bفرط نشاط الغدة الدرقية\b': 'مرض فرط نشاط الغدة الدرقية',
        r'\bقصور الغدة الدرقية\b': 'مرض قصور الغدة الدرقية',
    }

    # Normalize specific terms
    general_replacements= {
        r'\bالغدة الدرقية\b': 'مرض الغدة الدرقية',
        r'\bسكري\b': 'مرض السكري',
        r'\bالسكري\b': 'مرض السكري',
        r'\bانزيم ALT\b': 'ALT',
        r'\bجلطة\b': 'مرض الجلطة',
        r'\bالجلطة\b': 'مرض الجلطة',
        r'\bربو\b': 'مرض الربو',
        r'\bفقر الدم\b': 'مرض فقر الدم',
      
        r'\bنقص الصفائح الدموية\b': 'مرض نقص الصفائح الدموية',
        r'\bسكري الحمل\b': 'مرض سكري الحمل',
        r'\bاختبار الغلوكوز الصيامي\b': 'مرض اختبار الجلوكوز الصيامي',
        r'\bاختبار الجلوكوز الصيامي\b': 'مرض اختبار الجلوكوز الصيامي',
        r'\bقياس النشاط الكهربائي للقلب\b': 'مرض قياس النشاط الكهربائي للقلب',
        r'\bRestingECG\b': 'مرض قياس النشاط الكهربائي للقلب',
        r'\bاقصى معدل ضربات القلب\b': 'مرض اقصى معدل ضربات القلب',
        r'\bMaxHR\b': 'مرض اقصى معدل ضربات القلب',
        r'\bسكر الدم الصائم\b': 'مرض سكر الدم الصائم',
        r'\bFastingBS\b': 'مرض سكر الدم الصائم',
        r'\bاختبار وظائف الرئة\b': 'مرض اختبار وظائف الرئة',
        r'\bبروتين سي التفاعلي\b': 'مرض بروتين سي التفاعلي',
        r'\bCRP\b': 'مرض بروتين سي التفاعلي',
        r'\bالانسولين\b': 'مرض الانسولين',
        r'\bانسولين\b': 'مرض الانسولين',
        r'\bالكوليسترول الجيد\b': 'مرض الكوليسترول الجيد',
        r'\bكوليسترول جيد\b': 'مرض الكوليسترول الجيد',
        r'\bHDL\b': 'مرض الكوليسترول الجيد',
        r'\bالكوليسترول الضار\b': 'مرض الكوليسترول الضار',
        r'\bكوليسترول ضار\b': 'مرض الكوليسترول الضار',
        r'\bLDL\b': 'مرض الكوليسترول الضار',
        r'\bخلايا الدم البيضاء\b': 'مرض خلايا الدم البيضاء',
        r'\bWBC\b': 'مرض خلايا الدم البيضاء',
        r'\bضغط الدم الانبساطي\b': 'مرض ضغط الدم الانبساطي',
        r'\bDBP\b': 'مرض ضغط الدم الانبساطي',
        r'\bالهيماتوكريت\b': 'مرض الهيماتوكريت',
        r'\bهيماتوكريت\b': 'مرض الهيماتوكريت',
        r'\bhematocrit\b': 'مرض الهيماتوكريت',
        r'\bتركيز الهيموجلوبين في الكريات\b': 'مرض تركيز الهيموجلوبين في الكريات',
        r'\bتركيز هيموجلوبين في الكريات\b': 'مرض تركيز الهيموجلوبين في الكريات',
        r'\bMCHC\b': 'مرض تركيز الهيموجلوبين في الكريات',
        r'\bمتوسط الهيموجلوبين في الكريات\b': 'مرض متوسط الهيموجلوبين في الكريات',
        r'\bمتوسط هيموجلوبين في الكريات\b': 'مرض متوسط الهيموجلوبين في الكريات',
        r'\bMCH\b': 'مرض متوسط الهيموجلوبين في الكريات',
        r'\bمتوسط حجم الكريات\b': 'مرض متوسط حجم الكريات',
        r'\bMCV\b': 'مرض متوسط حجم الكريات',
        r'\bضغط الدم الانقباضي\b': 'مرض ضغط الدم الانقباضي',
        r'\bSBP\b': 'مرض ضغط الدم الانقباضي',
        r'\bالكرياتينين\b': 'مرض الكرياتينين',
        r'\bكرياتينين\b': 'مرض الكرياتينين',
        r'\bالهيموغلوبين السكري\b': 'مرض الهيموغلوبين السكري',
        r'\bهيموغلوبين سكري\b': 'مرض الهيموغلوبين السكري',
        r'\bالهيموغلوبين\b': 'مرض الهيموغلوبين السكري',
        r'\bهيموغلوبين\b': 'مرض الهيموغلوبين السكري',
        r'\bHbA1c\b': 'مرض الهيموغلوبين السكري',
        r'\bالتلاسيميا\b': 'مرض التلاسيميا',
        r'\bتلاسيميا\b': 'مرض التلاسيميا',
        r'\bالثلاسيميا\b': 'مرض التلاسيميا',
        r'\bثلاسيميا\b': 'مرض التلاسيميا',
        r'\bقلب\b': 'مرض القلب',
        r'\bالقلب\b': 'مرض القلب',
        r'\bزهايمر\b': 'مرض الزهايمر',
        r'\bالزهايمر\b': 'مرض الزهايمر',
        r'\bفرط نشاط الغدة الدرقية\b': 'مرض فرط نشاط الغدة الدرقية',
        r'\bقصور الغدة الدرقية\b': 'مرض قصور الغدة الدرقية',
        r'\bارتفاع ضغط الدم\b': 'مرض ارتفاع ضغط الدم',
        r'\bالتهاب الكبد\b': 'مرض التهاب الكبد',
        r'\bالربو\b': 'مرض الربو',
        r'\bالاكتئاب\b': 'مرض الاكتئاب',
        r'\bاكتئاب\b': 'مرض الاكتئاب',
        r'\bالباركنسون\b': 'مرض الباركنسون',
        r'\bباركنسون\b': 'مرض الباركنسون'
    }

    # استبدال الحالات الدقيقة أولًا
    for pattern, replacement in specific_replacements.items():
        text = re.sub(pattern, replacement, text)

        # استبدال الحالات العامة
    for pattern, replacement in general_replacements.items():
        text = re.sub(pattern, replacement, text)

    return text

# ======== دالة تنظيف المدخلات باستخدام NLTK ========
def preprocess_input(text):
    # تطبيع النص قبل التنظيف
    text = normalize_text(text)
    # تحويل النص إلى حروف صغيرة فقط دون حذف كلمات شائعة
    text = text.lower()
    # تحليل الكلمات
    tokens = nltk.word_tokenize(text)
    # استخدام التنصير فقط بدون حذف
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # إعادة تجميع النص
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# ======== إعداد دالة استخراج الـ Slots باستخدام Cosine Similarity ========
def extract_slots(user_input, x_list):
    # تنظيف المدخلات
    cleaned_input = preprocess_input(user_input)
    # تحويل المدخلات إلى متجه باستخدام النموذج
    # noinspection PyUnresolvedReferences
    user_input_vector = model.encode(cleaned_input)

    THRESHOLD = 0.75  # خفض العتبة للتشابه
    extracted_x = None
    max_similarity_x = 0

    # تحقق من التطابق الجزئي باستخدام `fuzzy matching`
    # أولوية للتطابق النصي الكامل
    for x in x_list:
        if preprocess_input(x) == cleaned_input:
            extracted_x = x
            max_similarity_x = 1  # تطابق مثالي
            break

    # إذا لم يتم العثور على تطابق كامل، استخدم التطابق الجزئي
    if not extracted_x:
        for x in x_list:
            match_score = fuzz.partial_ratio(x, user_input)
            if match_score > max_similarity_x and match_score >= THRESHOLD:
                max_similarity_x = match_score
                extracted_x = x
    print("Extracted X:", extracted_x)  # طباعة النتيجة المستخرجة لـ X

    # إذا لم يتم العثور على تطابق مباشر، استخدم cosine similarity
    if not extracted_x:
        for x in x_list:
            x_vector = model.encode(preprocess_input(x))
            similarity = cosine_similarity([user_input_vector], [x_vector])[0][0]

            if similarity > max_similarity_x and similarity >= THRESHOLD:
                max_similarity_x = similarity
                extracted_x = x

    print(f"Extracted X: {extracted_x} with similarity {max_similarity_x}")

    # استخراج نوع المعلومات المطلوبة
    extracted_info_type = None
    max_similarity_info = 0

    for intent in df['intents']:
        for pattern in intent['patterns']:
            # استبدال "[X]" بمحتوى extracted_x لأجل مطابقة دقيقة
            pattern_with_x = pattern.replace("[X]", extracted_x if extracted_x else "")
            pattern_cleaned = preprocess_input(pattern_with_x)
            # noinspection PyUnresolvedReferences
            pattern_vector = model.encode(pattern_cleaned)

            similarity = cosine_similarity([user_input_vector], [pattern_vector])[0][0]
            print(f"Checking pattern: {pattern_cleaned}, Similarity: {similarity}")

            if similarity > max_similarity_info and similarity >= THRESHOLD:
                max_similarity_info = similarity
                extracted_info_type = intent['tag']

    print(f"Extracted Info Type: {extracted_info_type} with similarity {max_similarity_info}")

    # التأكد من توفر نوع المعلومات والموضوع
    if extracted_x and extracted_info_type:
        return extracted_info_type, extracted_x
    else:
        return None, None

# ======== دالة الشات بوت باستخدام Slots ========
def chatbot(user_question):
    if not user_question.strip() or len(user_question) < 3:
        return "يرجى إدخال سؤال أو استفسار مناسب."

    # استخراج جميع الأنواع المتاحة في البيانات
    all_x = set()
    intent_types = {}
    for intent in df['intents']:
        if 'X' in intent['slots']:
            types = set(intent['slots']['X'])
            all_x.update(types)
            intent_types[intent['tag']] = types

    # استخراج نوع المعلومات والموضوع من السؤال
    info_type, x_name = extract_slots(user_question, all_x)

    print(f"Final Extracted Info Type: {info_type}, Final Extracted X: {x_name}")

    # التأكد من أن x_name ينتمي للـ info_type
    if info_type and x_name and x_name in intent_types.get(info_type, set()):
        for intent in df['intents']:
            if intent['tag'] == info_type:
                response = intent['responses'].get(x_name)
                if response:
                    return response
    return "عذرًا، لا أستطيع العثور على إجابة لهذا السؤال."

# ======== إعداد واجهة Flask ========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('question')

    if not user_input:
        return jsonify({"error": "يرجى إدخال سؤال."}), 400

    response = chatbot(user_input)
    return jsonify({"response": response})

# ======== تشغيل التطبيق ========
if __name__ == '__main__':
    load_model()  # تحميل النموذج هنا قبل تشغيل التطبيق
    app.run(debug=True)