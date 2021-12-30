from flask import * 
import json,response,cv2,os 

from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'static/uploads' 
DOWNLOAD_FOLDER='static/detections/'
app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')  
def upload():  
    return render_template("index.html")  
 
@app.route('/upload_image', methods = ['POST'])  
def upload_image():  
    if request.method == 'POST': 
        f = request.files['image']  
        fn=os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        f.save(fn)
        gn,dic=response.get_response(fn)
    return render_template("result.html", display_detection = DOWNLOAD_FOLDER+gn, fname = fn,dictionary=json.dumps(dic))  

if __name__ == '__main__':  
    app.run(debug = True) 