from googletrans import Translator
import json

def translate_thai_to_english(thai_text):
    translator = Translator()
    try:
        translation = translator.translate(thai_text, src='th', dest='en')
        return translation.text
    except Exception as e:
        print("An error occurred during translation:", e)
        return None

def translate_json_file(input_file, output_file):
    print("Translating...")
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    translated_data = []
    for item in json_data:
        thai_text = item[0]
        target = item[1]
        english_text = translate_thai_to_english(thai_text)
        if english_text is not None:
            translated_item = [english_text, target]
            translated_data.append(translated_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print("Translation saved to", output_file)

def main():
    input_file = "test.json"  # Replace with your input JSON file path
    output_file = "translated_output.json"  # Replace with your desired output JSON file path
    translate_json_file(input_file, output_file)

if __name__ == "__main__":
    main()
