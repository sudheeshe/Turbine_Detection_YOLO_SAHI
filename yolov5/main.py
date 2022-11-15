from detect import run
from sahi.predict import predict
from flask import Flask,request,render_template,redirect, url_for
import os
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)

# The image which gets through UI wille get saved at this location
#app.config["IMAGE_UPLOADS"] = "static/Images"

shutil.rmtree('runs/detect')
os.mkdir('runs/detect')

app.config["IMAGE_UPLOADS"] = "runs/detect"

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]


@app.route('/',methods = ["GET","POST"])
def upload_image():

	model = str(request.form.get('model'))
	size = str(request.form.get('size'))


	save_img = r'runs/detect'

	model_type = 'yolov5'

	model_path_small = 'saved_model/Yolo_Small.pt'
	model_config_path_small = 'customyolosmall.yaml'

	model_path_X = 'saved_model/Yolo_X.pt'
	model_config_path_X = 'customyolov5x.yaml'

	if request.method == "POST":

		shutil.rmtree('static/Images/exp')
		os.mkdir('static/Images/exp')

		image = request.files['file']

		filename = secure_filename(image.filename)

		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))

		if model=='yolo_small':
			run(weights=model_path_small,
				source=os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename),
				project='static/Images/exp',
				name='visuals',
				conf_thres=0.3,
				iou_thres=0.567)


		elif model=='yolo_x':
			run(weights=model_path_X,
				source=os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename),
				project='static/Images/exp',
				name='visuals',
				conf_thres=0.3,
				iou_thres=0.417)



		elif model == 'yolo_small_sahi':
			shutil.rmtree('static/Images/exp')
			predict(source=os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename),
					slice_height=int(size),
					slice_width=int(size),
					model_type=model_type,
					model_path=model_path_small,
					model_config_path=model_config_path_small,
					model_confidence_threshold=0.3,
					postprocess_match_threshold=0.567,
					project='D:/Ineuron/Project_workshop/Turbine_Detection_Inferencing_SAHI/yolov5/static/Images',
					name='exp',
					visual_export_format=filename.split('.')[-1])

		elif model == 'yolo_x_sahi':
			shutil.rmtree('static/Images/exp')
			predict(source=os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename),
					slice_height=int(size),
					slice_width=int(size),
					model_type=model_type,
					model_path=model_path_X,
					model_config_path=model_config_path_X,
					model_confidence_threshold=0.3,
					postprocess_match_threshold=0.417,
					project='D:/Ineuron/Project_workshop/Turbine_Detection_Inferencing_SAHI/yolov5/static/Images',
					name='exp',
					visual_export_format=filename.split('.')[-1])



		return render_template("index.html", filename=filename)



	return render_template('index.html')



@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename = "/Images/exp/visuals" + "prediction_" + filename), code=301)


app.run(debug=True,port=2000)




