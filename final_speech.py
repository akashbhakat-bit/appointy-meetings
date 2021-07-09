import os
from flask_cors import CORS
from flask import Flask, render_template, request, redirect
from pydub import AudioSegment
from pydub.utils import make_chunks


def main_sql(file):
  import librosa
  import tensorflow as tf
  import numpy as np

  SAVED_MODEL_PATH = "sql_model_94.h5"
  SAMPLES_TO_CONSIDER = 66150

  class _Keyword_Spotting_Service:
      

      model = None
      _mapping = [
          "issuer_fee_amount",
          "acquirer_settlement_amount",
          "cardholder_billing_amount",
          "cashback_count",
          "transaction_count",
          "cashback_amount",
          "surcharge_amount",
          "transaction_amount",
          "issuer_settlement_amount",
          "merchant_transaction_amount"
      ]
      _instance = None


      def predict(self, file_path):
          

          # extract MFCC
          MFCCs = self.preprocess(file_path)

          # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
          MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

          # get the predicted label
          predictions = self.model.predict(MFCCs)
          predicted_index = np.argmax(predictions)
          predicted_keyword = self._mapping[predicted_index]
          return predicted_keyword


      def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
          

          # load audio file
          signal, sample_rate = librosa.load(file_path)

          if len(signal) >= SAMPLES_TO_CONSIDER:
              # ensure consistency of the length of the signal
              signal = signal[:SAMPLES_TO_CONSIDER]

              # extract MFCCs
          MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                          hop_length=hop_length)
          return MFCCs.T


  def Keyword_Spotting_Service():
      

      # ensure an instance is created only the first time the factory function is called
      if _Keyword_Spotting_Service._instance is None:
          _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
          _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
      return _Keyword_Spotting_Service._instance




  if __name__ == "__main__":

      # create 2 instances of the keyword spotting service
      kss = Keyword_Spotting_Service()
      kss1 = Keyword_Spotting_Service()

      # check that different instances of the keyword spotting service point back to the same object (singleton)
      assert kss is kss1

      # make a prediction
      #keyword = kss.predict("/content/drive/MyDrive/Custom Speech/dataset/cashback_count/Copy of Copy of output_10.wav")
      keyword=kss.predict(file)

      print(keyword)
      return keyword

def bank(file):
  import librosa
  import tensorflow as tf
  import numpy as np

  SAVED_MODEL_PATH = "bank_model.h5"
  SAMPLES_TO_CONSIDER = 66150

  class _Keyword_Spotting_Service:
      

      model = None
      _mapping = [
          "qiwi_bank",
          "far_eastern_bank",
          "alfa_bank",
          "west_siberian_commercial_bank",
          "ross_bank"
      ]
      _instance = None


      def predict(self, file_path):
          

          # extract MFCC
          MFCCs = self.preprocess(file_path)

          # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
          MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

          # get the predicted label
          predictions = self.model.predict(MFCCs)
          predicted_index = np.argmax(predictions)
          predicted_keyword = self._mapping[predicted_index]
          return predicted_keyword


      def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
          

          # load audio file
          signal, sample_rate = librosa.load(file_path)

          if len(signal) >= SAMPLES_TO_CONSIDER:
              # ensure consistency of the length of the signal
              signal = signal[:SAMPLES_TO_CONSIDER]

              # extract MFCCs
              MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                          hop_length=hop_length)
          return MFCCs.T


  def Keyword_Spotting_Service():
      

      # ensure an instance is created only the first time the factory function is called
      if _Keyword_Spotting_Service._instance is None:
          _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
          _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
      return _Keyword_Spotting_Service._instance




  if __name__ == "__main__":

      # create 2 instances of the keyword spotting service
      kss = Keyword_Spotting_Service()
      kss1 = Keyword_Spotting_Service()

      # check that different instances of the keyword spotting service point back to the same object (singleton)
      assert kss is kss1

      # make a prediction
      #keyword = kss.predict("/content/drive/MyDrive/CustomBank/dataset/alfa_bank/Copy of output_alfa_11.wav")
      keyword=kss.predict(file)
      print(keyword)
      return keyword

def month(file):
  import librosa
  import tensorflow as tf
  import numpy as np

  SAVED_MODEL_PATH = "month_model.h5"
  SAMPLES_TO_CONSIDER = 66150

  class _Keyword_Spotting_Service:
      

      model = None
      _mapping = [
          "july",
          "june"
      ]
      _instance = None


      def predict(self, file_path):
          

          # extract MFCC
          MFCCs = self.preprocess(file_path)

          # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
          MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

          # get the predicted label
          predictions = self.model.predict(MFCCs)
          predicted_index = np.argmax(predictions)
          predicted_keyword = self._mapping[predicted_index]
          return predicted_keyword


      def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
          

          # load audio file
          signal, sample_rate = librosa.load(file_path)

          if len(signal) >= SAMPLES_TO_CONSIDER:
              # ensure consistency of the length of the signal
              signal = signal[:SAMPLES_TO_CONSIDER]

              # extract MFCCs
              MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                          hop_length=hop_length)
          return MFCCs.T


  def Keyword_Spotting_Service():
      

      # ensure an instance is created only the first time the factory function is called
      if _Keyword_Spotting_Service._instance is None:
          _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
          _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
      return _Keyword_Spotting_Service._instance




  if __name__ == "__main__":

      # create 2 instances of the keyword spotting service
      kss = Keyword_Spotting_Service()
      kss1 = Keyword_Spotting_Service()

      # check that different instances of the keyword spotting service point back to the same object (singleton)
      assert kss is kss1

      # make a prediction
      #keyword = kss.predict("/content/drive/MyDrive/CustomMonth/dataset/june/Copy of output_june_12.wav")
      keyword=kss.predict(file)
      print(keyword)
      return keyword

def year(file):
  import librosa
  import tensorflow as tf
  import numpy as np

  SAVED_MODEL_PATH = "year_model.h5"
  SAMPLES_TO_CONSIDER = 66150

  class _Keyword_Spotting_Service:
      

      model = None
      _mapping = [
          "2018",
          "2019",
          "2021",
          "2020"
      ]
      _instance = None


      def predict(self, file_path):
          

          # extract MFCC
          MFCCs = self.preprocess(file_path)

          # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
          MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

          # get the predicted label
          predictions = self.model.predict(MFCCs)
          predicted_index = np.argmax(predictions)
          predicted_keyword = self._mapping[predicted_index]
          return predicted_keyword


      def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
          

          # load audio file
          signal, sample_rate = librosa.load(file_path)

          if len(signal) >= SAMPLES_TO_CONSIDER:
              # ensure consistency of the length of the signal
              signal = signal[:SAMPLES_TO_CONSIDER]

              # extract MFCCs
              MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                          hop_length=hop_length)
          return MFCCs.T


  def Keyword_Spotting_Service():
      

      # ensure an instance is created only the first time the factory function is called
      if _Keyword_Spotting_Service._instance is None:
          _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
          _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
      return _Keyword_Spotting_Service._instance




  if __name__ == "__main__":

      # create 2 instances of the keyword spotting service
      kss = Keyword_Spotting_Service()
      kss1 = Keyword_Spotting_Service()

      # check that different instances of the keyword spotting service point back to the same object (singleton)
      assert kss is kss1

      # make a prediction
      #keyword = kss.predict("/content/drive/MyDrive/CustomYear/dataset/2020/Copy of Copy of output_2020_1.wav")
      keyword=kss.predict(file)
      print(keyword)
      return keyword

def product(file):
  import librosa
  import tensorflow as tf
  import numpy as np

  SAVED_MODEL_PATH = "product_model_97.h5"
  SAMPLES_TO_CONSIDER = 66150

  class _Keyword_Spotting_Service:
      

      model = None
      _mapping = [
          "prepaid",
          "deferred_debit",
          "debit",
          "credit",
          "charge"
      ]
      _instance = None


      def predict(self, file_path):
          

          # extract MFCC
          MFCCs = self.preprocess(file_path)

          # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
          MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

          # get the predicted label
          predictions = self.model.predict(MFCCs)
          predicted_index = np.argmax(predictions)
          predicted_keyword = self._mapping[predicted_index]
          return predicted_keyword


      def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
          

          # load audio file
          signal, sample_rate = librosa.load(file_path)

          if len(signal) >= SAMPLES_TO_CONSIDER:
              # ensure consistency of the length of the signal
              signal = signal[:SAMPLES_TO_CONSIDER]

              # extract MFCCs
              MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                          hop_length=hop_length)
          return MFCCs.T


  def Keyword_Spotting_Service():
      

      # ensure an instance is created only the first time the factory function is called
      if _Keyword_Spotting_Service._instance is None:
          _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
          _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
      return _Keyword_Spotting_Service._instance




  if __name__ == "__main__":

      # create 2 instances of the keyword spotting service
      kss = Keyword_Spotting_Service()
      kss1 = Keyword_Spotting_Service()

      # check that different instances of the keyword spotting service point back to the same object (singleton)
      assert kss is kss1

      # make a prediction
      #keyword = kss.predict("/content/drive/MyDrive/CustomProductType/dataset/charge/Copy of Copy of new_charge_output_1.wav")
      keyword=kss.predict(file)
      print(keyword)
      return keyword

def voice(file):
  import librosa
  import tensorflow as tf
  import numpy as np

  SAVED_MODEL_PATH = "voice_identifier_model_old.h5"
  SAMPLES_TO_CONSIDER = 66150

  class _Keyword_Spotting_Service:
      

      model = None
      _mapping = [
        "bharath",
          "akash",
          "puneeth"
      ]
      _instance = None


      def predict(self, file_path):
          

          # extract MFCC
          MFCCs = self.preprocess(file_path)

          # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
          MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

          # get the predicted label
          predictions = self.model.predict(MFCCs)
          predicted_index = np.argmax(predictions)
          predicted_keyword = self._mapping[predicted_index]
          return predicted_keyword


      def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
          

          # load audio file
          signal, sample_rate = librosa.load(file_path)

          if len(signal) >= SAMPLES_TO_CONSIDER:
              # ensure consistency of the length of the signal
              signal = signal[:SAMPLES_TO_CONSIDER]

              # extract MFCCs
          MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                          hop_length=hop_length)
          return MFCCs.T


  def Keyword_Spotting_Service():
      

      # ensure an instance is created only the first time the factory function is called
      if _Keyword_Spotting_Service._instance is None:
          _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
          _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
      return _Keyword_Spotting_Service._instance




  if __name__ == "__main__":

      # create 2 instances of the keyword spotting service
      kss = Keyword_Spotting_Service()
      kss1 = Keyword_Spotting_Service()

      # check that different instances of the keyword spotting service point back to the same object (singleton)
      assert kss is kss1

      # make a prediction
      keyword = kss.predict(file)
      print(keyword)
      return keyword




app = Flask(__name__)

CORS(app)

@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        result=""
        print("FORM DATA RECEIVED")
        
        if "file" not in request.files:
            print("Case 1")
            return redirect(request.url)

        file = request.files["file"]
        
        if file:
            file_name="input"
            file.save(file_name)
            
            myaudio = AudioSegment.from_file(file_name , "wav") 
            chunk_length_ms = 3000 # pydub calculates in millisec
            chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of 3 sec

            #Export all of the individual chunks as wav files

            for i, chunk in enumerate(chunks):
                chunk_name = "chunk{0}.wav".format(i)
                print("exporting", chunk_name)
                chunk.export(chunk_name, format="wav")
                
                
            x=[]
            speaker=[]
            if(os.path.isfile("chunk0.wav")):
                x.append(str(main_sql('chunk0.wav'))+" ")
                speaker.append(str(voice('chunk0.wav')))
                os.remove("chunk0.wav")

            if(os.path.isfile("chunk1.wav")):
                x.append(str(bank('chunk1.wav'))+" ")
                os.remove("chunk1.wav")

            if(os.path.isfile("chunk2.wav")):
                x.append(str(month('chunk2.wav'))+" ")
                os.remove("chunk2.wav")

            if(os.path.isfile("chunk3.wav")):
                x.append(str(year('chunk3.wav'))+" ")
                os.remove("chunk3.wav")

            if(os.path.isfile("chunk4.wav")):
                x.append(str(product('chunk4.wav'))+" ")
                os.remove("chunk4.wav")

            b=1
            y=len(x)
            i=0
            while(b<=y):
                a=x[i].split("_")
                for k in a:
                    #print("*")
                    print(k)
                    x.append(k)
                    del x[i]
                    b+=1

            
            print(x)
            dictionary={'input':x,'speaker':speaker}
            print(dictionary)
            return dictionary
            os.remove(file_name)
            '''
            result+=transcript(input)
            print(result)
            os.remove(input)
            '''
            

            



if __name__ == "__main__":
    app.run()