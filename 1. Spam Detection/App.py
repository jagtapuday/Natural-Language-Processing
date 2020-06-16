from flask import Flask,request,jsonify
from Prediction import *
import json
import pandas as pd
app=Flask(__name__)


@app.route('/predict', methods=['GET','POST'])
def request_url():
    print("I am in Prediction")


    try:

        if request.method == 'GET':
            Name = str(request.args.get('Name'))
            return jsonify({"Status": "The Message is {}"})
        else:
            Data = request.data
            dataDict = json.loads(Data)
            print("Data->", dataDict, type(dataDict))

            txt = dataDict['txt']
            demo = pd.Series(txt)
            print("Message->{}".format(demo))

            demo = Preprocessing(demo)

            predict_features1 = find_features(demo[0])

            return jsonify({
                "Data" : str(classifier.classify_many(predict_features1)[0]),
                "Status": "The Message is {}".format('not spam' if classifier.classify_many(predict_features1)[0] == 0 else 'spam')
            })

    except Exception as e:
        print("Error----->", e)
        return jsonify({
            "Status": "Failed"
        })
    finally:
        print("Done")


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)