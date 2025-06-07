from googletrans import Translator

class Translate:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text, language_og = "auto", language_des = "en"):
        try:
            result = self.translator.translate(text, src= language_og, dest= language_des)
            return result.text
        except Exception as e:
            print(e)
            return None
    
    def masive_translate(self, list, language_og = "auto", language_des = "en"):
        results = []
        try:
            for text in list:
                results.append(self.translate(text, language_og, language_des))
            return results
        except Exception as e:
            print(e)
            return None