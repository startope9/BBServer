from flask import Flask, request, send_from_directory, session
from flask_cors import CORS
import os
import shutil
import cv2
import numpy as np
import pyrebase
from flask_session import Session
from ultralytics import YOLO
import time
import imageio
import pandas as pd
import json
import redis


app = Flask(__name__)


#Session

SECRET_KEY = 'afuibewyhfqwj9028yr378y'
SESSION_TYPE = 'redis'
SESSION_REDIS = redis.from_url(os.environ.get('SESSION_REDIS'))
app.config.from_object(__name__)
app.config['SESSION_TYPE'] = SESSION_TYPE
app.config['SESSION_REDIS'] = SESSION_REDIS
app.config['SECRET_KEY'] = SECRET_KEY
sess = Session()
sess.init_app(app)


# Session(app)
CORS(app, supports_credentials=True, origins=r'https://ball-badminton.vercel.app/*')

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


firebase_credentials = os.environ.get('FIREBASE_CREDENTIALS_JSON')
config = json.loads(firebase_credentials)

firebase = pyrebase.initialize_app(config)
db = firebase.database()


@app.route('/')
def home():
    return "I'm home!"


@app.route('/login/<int:val>', methods=['POST'])
def letsLogin(val):

    # 1 for coach
    if val==1:
        req = request.get_json()
        ref = db.child('Coach').get()

        for key, value in ref.val().items():
            if value['Name'] == req['Name'] and value['Pass'] == req['Pass']:
                session['user'] = value['Name']
                session['user_job'] = 'Coach'
                return '200'

    # 2 for umpire
    if val==2:

        req = request.get_json()
        ref = db.child('Umpire').get()

        for key, value in ref.val().items():
            if value['Name'] == req['Name'] and value['Pass'] == req['Pass']:
                session['user'] = value['Name']
                session['user_job'] = 'Umpire'
                return '300'
            
    return '500'

# return 500 if user not in session
@app.route('/team/<string:num>', methods=['POST'])
def teamDetails(num):

    if 'user' in session:

        if num == '1':
            req = request.get_json()
            match_num = session.get('match')

            db.child('match').child(match_num).child('team1').push(req)
            db.child('match').child(match_num).push({'status':1})

            return '200'
        

        if num == '2': 
            req = request.get_json()
            match_num = session.get('match')
            db.child('match').child(match_num).child('team2').push(req)

            return '200'

    return '500'



@app.route('/number', methods=['POST'])
def createTree():

    if 'user' in session:

        user = session.get('user')
        req = str(request.get_json()['number'])
        if db.child('match').child('match'+req).get().val() is not None:
            return '300'
        else:
            session['match'] = 'match'+req

            db.child('match').child('match'+req).push({'umpire':user})

            return '200'


    return '500'



@app.route('/score/<string:teamno>', methods=['POST'])
def addScore(teamno):

    if 'user' in session:

        if teamno == '1':
            req = request.get_json()
            match_num = session.get('match')

            ref = db.child('match').child(match_num).get()
            refbranch = db.child('match').child(match_num).child('team1').get()


            for key, value in ref.val().items():

                try:
                    if value['status'] == 1:
                        for k, val in refbranch.val().items():
                            db.child('match').child(match_num).child('team1').child(k).update({'score':req['score']})

                except:
                    continue


            return '200'

        if teamno == '2':
            req = request.get_json()
            match_num = session.get('match')

            ref = db.child('match').child(match_num).get()
            refbranch = db.child('match').child(match_num).child('team2').get()


            for key, value in ref.val().items():

                try:
                    if value['status'] == 1:
                        for k, val in refbranch.val().items():
                            db.child('match').child(match_num).child('team2').child(k).update({'score':req['score']})

                except:
                    continue


            return '200'

    return '500'


@app.route('/getLiveScore', methods=['POST'])
def liveScore():

    ref = db.child('match').get()
    
    arr = []
    maxMatch = 0

    for key, value in ref.val().items():

        maxMatch = int(key[5:]) if maxMatch < int(key[5:]) else maxMatch
        
        for k, val in db.child('match').child(key).get().val().items():
            try:
                if val['status']:
                
                    for ke, val in db.child('match').child(key).child('team1').get().val().items():
                        arr.append(val['score'])        # 0 - team1 score
                        arr.append(val['TeamName'])     # 1 - team1 Name
                        try:
                            val['Head']                            
                            arr.append(val['Head'][len(val['Head'])-1])     #2 team1 head  
                        except:
                            arr.append(1)

                    for ke, val in db.child('match').child(key).child('team2').get().val().items():
                        arr.append(val['score'])        # 3 - team2 score
                        arr.append(val['TeamName'])     # 4 - team1 Name
                        try:
                            val['Head']
                            arr.append(val['Head'][len(val['Head'])-1])     #5 team1 head
                        except:
                            arr.append(1)
                    break

            except:
                continue

    
    if not len(arr):

        for key, value in db.child('match').child('match'+str(maxMatch)).child('team1').get().val().items():
            arr.append(value['score'])
            arr.append(value['TeamName'])
        arr.append(0)
        for key, value in db.child('match').child('match'+str(maxMatch)).child('team2').get().val().items():
            arr.append(value['score'])
            arr.append(value['TeamName'])


        return arr

    else:
        # probability calculation 
        if arr[0] != 0 or arr[3] != 0:
            
            arr.append(round((arr[0]/(arr[0]+arr[3]))*100, 2))
            arr.append(round((arr[3]/(arr[0]+arr[3]))*100, 2))
        else:
 
            team1 = []
            team2 =  []
            team1points = []
            team2points = []
            winner = []

            for key, value in ref.val().items():
                
                a=0
                b=0
                c = ''
                d = ''

                for k, val in db.child('match').child(key).get().val().items():

                    try:
                        if not val['status']:

                            for i, j in db.child('match').child(key).child('team1').get().val().items():
                                # print('team1', j['score'])
                                team1.append(j['TeamName'])     #0  
                                team1points.append(j['score'])  #1
                                a = j['score']
                                c = j['TeamName']


                            for i, j in db.child('match').child(key).child('team2').get().val().items():
                                # print('team2', j['score'])
                                team2.append(j['TeamName'])     #2
                                team2points.append(j['score'])  #3
                                b = j['score']
                                d = j['TeamName']
                                winner.append(c if a>b else d)

                    except:
                        continue

            data = {
            "team1" : team1,
            "team2" : team2,
            "team1_points": team1points,
            "team2_points":team2points,
            "winner":winner
            }

            df = pd.DataFrame(data)

            all_teams=set()
            for i in df['team1']:
                all_teams.add(i)
            for i in df['team2']:
                all_teams.add(i)
            team1=arr[1].upper()
            team2=arr[4].upper()
            if team1 in all_teams and team2 in all_teams:
                total_team1=0
                total_team2=0
                df = df[((df['team1'] == team1) & (df['team2'] == team2)) | ((df['team1'] == team2) & (df['team2'] == team1))]
                number_of_matches=len(df)

                if number_of_matches == 0:
                    arr.append(50)
                    arr.append(50)

                else:
                    total_team1_recent=0
                    total_team2_recent=0
                    # for i in range(0,number_of_matches):
                    for j in df['winner']:
                        if(j==team1):
                            total_team1+=1
                    total_team2=number_of_matches-total_team1
                    if(number_of_matches>5):
                        df1=df.tail()
                        number_of_matches_recent=5
                        for i in range(0,5):
                            for j in df['winner']:
                                if(j==team1):
                                    total_team1_recent+=1
                        total_team2_recent=number_of_matches_recent-total_team1_recent
                    team1_win=0
                    team2_win=0
                    if(number_of_matches>5):
                        team1_win=(0.5*total_team1/number_of_matches+0.5*total_team1_recent/5)*100
                        if(team1_win==0):
                            team1_win=1
                        if(team1_win==100):
                            team1_win=99
                        team2_win=100-team1_win
                        arr.append(round(team1_win, 2))
                        arr.append(round(team2_win, 2))
                    else:
                        team1_win=(total_team1/number_of_matches)*100
                        if(team1_win==0):
                            team1_win=1
                        if(team1_win==100):
                            team1_win=99
                        team2_win=100-team1_win
                        arr.append(round(team1_win, 2))
                        arr.append(round(team2_win, 2))
            else:
                arr.append(50)
                arr.append(50)


        return arr



@app.route('/api/upload/<int:choice>', methods=['POST'])
def upload_video(choice):

    if 'user' in session:

        if 'video' not in request.files:
            return 'No file part', 400

        video_file = request.files['video']

        if video_file.filename == '':
            return 'No selected file', 400
        

        # speed track - optical flow
        if choice == 1:

            if video_file:
                user = session.get('user')
                new_path = app.config['UPLOAD_FOLDER']+'/'+f'{user}/one'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                filename = os.path.join(new_path, video_file.filename)
                video_file.save(filename)
                
                # Perform video processing using OpenCV
                proc_path = app.config['PROCESSED_FOLDER']+f'/{user}/one'
                if not os.path.exists(proc_path):
                    os.makedirs(proc_path)
                processed_filename = os.path.join(proc_path, 'processed_' + video_file.filename)
                process_video_speed_track(filename, processed_filename)

                return 'Video uploaded and processed successfully'
            
        # ball track - ball tracking    
        elif choice == 2:

            if video_file:
                user = session.get('user')
                new_path = app.config['UPLOAD_FOLDER']+'/'+f'{user}/two'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                filename = os.path.join(new_path, video_file.filename)
                video_file.save(filename)
                
                # Perform video processing using OpenCV
                
                proc_path = app.config['PROCESSED_FOLDER']+f'/{user}/two'
                if not os.path.exists(proc_path):
                    os.makedirs(proc_path)
                processed_filename = os.path.join(proc_path, 'processed_' + video_file.filename)
                process_video_ball_track(filename, processed_filename)

                return 'Video uploaded and processed successfully'
            
        # Pose Estimation - markerless    
        elif choice == 3:
            if video_file:
                
                user = session.get('user')
                new_path = app.config['UPLOAD_FOLDER']+'/'+f'{user}/three'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                filename = os.path.join(new_path, video_file.filename)
                video_file.save(filename)
                
                # Perform video processing using OpenCV
                
                proc_path = app.config['PROCESSED_FOLDER']+f'/{user}/three'
                if not os.path.exists(proc_path):
                    os.makedirs(proc_path)
                processed_filename = os.path.join(proc_path, 'processed_' + video_file.filename)
                process_video_markerless(filename, processed_filename)

                return 'Video uploaded and processed successfully'
            
        elif choice == 4:
            if video_file:
                
                user = session.get('user')
                new_path = app.config['UPLOAD_FOLDER']+'/'+f'{user}/four'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                filename = os.path.join(new_path, video_file.filename)
                video_file.save(filename)
                
                # Perform video processing using OpenCV
                
                proc_path = app.config['PROCESSED_FOLDER']+f'/{user}/four'
                if not os.path.exists(proc_path):
                    os.makedirs(proc_path)
                processed_filename = os.path.join(proc_path, 'processed_' + video_file.filename)
                process_video_court_player(filename, processed_filename)

                return 'Video uploaded and processed successfully'
            
    
    return '500'

            


def process_video(input_filename, output_filename):
    cap = cv2.VideoCapture(input_filename)

    out = cv2.VideoWriter(output_filename, cv2.CAP_OPENCV_MJPEG, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Write the grayscale frame to the output video
        out.write(grayscale_frame)

    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_video_ball_track(input_filename, output_filename):
    cap = cv2.VideoCapture(input_filename)

    out = cv2.VideoWriter(output_filename, cv2.CAP_OPENCV_MJPEG, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

    # Convert the frame to HSV color space for better color detection
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the yellow color range
    # Adjust these values as needed   137, 133, 57 205, 215, 35
    # lower_yellow = np.array([100, 170, 170])
        lower_yellow = np.array([30, 185, 185])

    # Adjust these values as needed    200, 204, 45 215, 220, 40
    # upper_yellow = np.array([130, 200, 200])
        upper_yellow = np.array([40, 220, 215])

    # Create a mask that isolates the yellow color in the frame
    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.inRange(src=frame, lowerb=lower_yellow, upperb=upper_yellow)

        # cv2.imshow('masked', mask)
    # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('hsv', hsv)

        for contour in contours:
        # Calculate the center and radius of the detected yellow object
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))

        # Only consider objects with a minimum radius to filter out noise
            if radius:
            # Draw a red dot on the frame and save it as an image
                cv2.circle(frame, center, 10, (0, 0, 255), -1)  # Draw a red dot
                # Write the grayscale frame to the output video
                out.write(frame)
                # break

        

    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# speed track - optical flow
def process_video_speed_track(input_filename, output_filename):

    cap = cv2.VideoCapture(input_filename)

    out = cv2.VideoWriter(output_filename, cv2.CAP_OPENCV_MJPEG, 30.0, (int(cap.get(3)), int(cap.get(4))))

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Create random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take the first frame and find corners in it
    ret, old_frame = cap.read()
    if ret:
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        line_duration = 1.5  # Time in seconds to display flow lines
        start_time = cv2.getTickCount()

        # Opacity (alpha) for lines
        line_opacity = 0.5

        # Keep track of the last 5 tracks
        last_tracks = []

        while True:
            # Read the new frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Initialize the Y-coordinate for text
            text_y = 30  # Starting position for text

            # Keep track of the last 5 tracks
            last_tracks.append(good_new)
            if len(last_tracks) > 5:
                last_tracks.pop(0)

            for j, track in enumerate(last_tracks):
                text_y = 20  # Adjust Y-coordinate for the next line
                if j == 4:  # Display speeds for the first 5 tracks
                    for i, (new, old) in enumerate(zip(track, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        delta_x = a - c
                        delta_y = b - d
                        distance_px = np.sqrt(delta_x ** 2 + delta_y ** 2)
                        distance_m = (distance_px / line_duration) / 100  # Assuming 1 pixel = 1 cm
                        speed_m_per_s = distance_m / line_duration  # Convert to m/s
                        speed_text = f"Player {i + 1}: {speed_m_per_s:.2f} m/s"

                        # Draw the tracks
                        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)

                        # Display speed text on the frame
                        cv2.putText(frame, speed_text, (10 + i * 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display the frame with reduced line opacity
            img = cv2.addWeighted(frame, 1, mask, line_opacity, 0)
            # cv2.imshow("frame", img)
            out.write(img)

            # Check if the line duration has elapsed, clear the mask
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            if elapsed_time >= line_duration:
                mask = np.zeros_like(old_frame)
                start_time = current_time

            k = cv2.waitKey(25) & 0xFF
            if k == 27:
                break

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)


    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Pose Estimation - markerless    
def process_video_markerless(input_filename, output_filename):

    cap = cv2.VideoCapture(input_filename)

    # out = cv2.VideoWriter(output_filename, cv2.CAP_OPENCV_MJPEG, 30.0, (int(cap.get(3)), int(cap.get(4))))


    model = YOLO('yolov8n-pose.pt')
    # video_path = "video.mp4"
    # cap = cv2.VideoCapture(video_path)
    writer = imageio.get_writer(output_filename, mode="I")
    while cap.isOpened():
    # Read a frame from the video
        success, frame = cap.read()

        if success:
            start_time = time.time()
            # Run YOLOv8 inference on the frame
            results = model(frame)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            # print("FPS :", fps)
            
            cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
            
            # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", annotated_frame)

            
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(annotated_frame)
            
            # Break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF ==ord('q'):
            #     break
        else:
            # Break the loop if the end of the video is reached
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

#aditi-add
def process_video_court_player(input_filename, output_filename):

    cap = cv2.VideoCapture(input_filename)

    out = cv2.VideoWriter(output_filename, cv2.CAP_OPENCV_MJPEG, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Write the grayscale frame to the output video
        out.write(grayscale_frame)

    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


@app.route('/api/get_processed_video/<string:user>/<int:choice>/<string:filename>')
def get_processed_video(user, choice, filename):
    if choice ==1:
        return send_from_directory(app.config['PROCESSED_FOLDER']+f'/{user}/one', filename)
    if choice ==2:
        return send_from_directory(app.config['PROCESSED_FOLDER']+f'/{user}/two', filename)
    if choice ==3:
        return send_from_directory(app.config['PROCESSED_FOLDER']+f'/{user}/three', filename)
    if choice ==4:
        return send_from_directory(app.config['PROCESSED_FOLDER']+f'/{user}/four', filename)



@app.route('/remove_dir', methods=['POST'])
def remove_the_dirs():

    if 'user' in session:
        user  = session.get('user')

        if os.path.exists(app.config['PROCESSED_FOLDER']+f'/{user}'):
            shutil.rmtree(app.config['PROCESSED_FOLDER']+f'/{user}')
        
        if os.path.exists(app.config['UPLOAD_FOLDER']+f'/{user}'):
            shutil.rmtree(app.config['UPLOAD_FOLDER']+f'/{user}')

        return '200'

    return '500'

@app.route('/getSessionInfo', methods=['POST'])
def returnSession():
    if 'user' in session:
        return [session['user'], session['user_job']]
    else:
        return []
    


@app.route('/uploadHead', methods=['POST'])
def uploadHead():


    if 'user' in session:

        req = request.get_json()
        match_num = session.get('match')


        ref = db.child('match').child(match_num).get()

        if req['team'] == 1:

            for key, value in ref.val().items():

                try:
                    if value['status'] == 1:

                        for k, val in db.child('match').child(match_num).child('team1').get().val().items():
                            db.child('match').child(match_num).child('team1').child(k).update({'Head':req['head1']})

                except:
                    continue

        
            return '200'

        elif req['team'] == 2:

            for key, value in ref.val().items():

                try:
                    if value['status'] == 1:

                        for k, val in db.child('match').child(match_num).child('team2').get().val().items():
                            db.child('match').child(match_num).child('team2').child(k).update({'Head':req['head2']})

                except:
                    continue

            return '200'
    
    return '500'



@app.route('/finish_match', methods=['POST'])
def matchFinish():

    if 'user' in session:

        match_num = session.get('match')

        ref = db.child('match').child(match_num).get()

        for key, value in ref.val().items():

            try:
                if value['status']:
                    db.child('match').child(match_num).child(key).update({'status':0})
            except:
                continue


        session.pop('match', None)


        return '200'
    
    return '500'



@app.route('/getAllMatches', methods=['POST'])
def returnAllMatches():


    if 'user' in session:

        ref = db.child('match').get()

                
        arr = []

        for key, value in ref.val().items():
            subarr = []
            
            subarr.append(key)
            for k, val in db.child('match').child(key).child('team1').get().val().items():
                try:
                    subarr.append(val['TeamName'])
                    subarr.append(val['score'])
                except:
                    continue
            for k, val in db.child('match').child(key).child('team2').get().val().items():
                try:
                    subarr.append(val['TeamName'])
                    subarr.append(val['score'])
                except:
                    continue

            arr.append(subarr)

        return arr

    return '500'

@app.route('/logout', methods=['POST'])
def logout():

    if 'user' in session:
        session.pop('user', None)
        session.pop('user_job', None)

    return '200'

if __name__ == "__main__":
    app.run(debug=True)

