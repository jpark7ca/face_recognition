import numpy as np
import dlib
from pkg_resources import resource_filename
import cv2
from threading import Thread


# FaceRecognition static class

class FaceRecognition:

    # image tuple: (left, top, right, bottom)

    @staticmethod
    def trim_bounds(bbox, image_shape):
        return max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], image_shape[1]), min(bbox[3], image_shape[0])

    @staticmethod
    def face_locations(image, upsample=1):
        face_detector = dlib.get_frontal_face_detector()  # use HOG
        # number of times to upsample = 1
        # face_detector returns dlib.fhog_object_detector which returns dlib.rectangles == face

        _ret = []
        for face in face_detector(image, upsample):
            _ret.append(FaceRecognition.trim_bounds((face.left(), face.top(), face.right(), face.bottom()), image.shape))

        return _ret

    @staticmethod
    def load_image(file, pixeltype=cv2.IMREAD_COLOR):
        _image = cv2.imread(file, pixeltype)
        return np.array(_image)

    @staticmethod
    def face_encodings(image, locations=None, upsample=1, jitter=1):
        # Generate the face encodings
        if locations is None:
            face_detector = dlib.get_frontal_face_detector()  # use HOG
            _raw_face_locations = face_detector(image, upsample)  # returns dlib *** RECT *** objects
        else:
            #  left: location[0], top: location[1], right: location[2], bottom: location[3]
            _raw_face_locations = [dlib.rectangle(location[0], location[1], location[2], location[3]) for location in locations]

        # small 5 points landmarks
        predictor_5_model_location = resource_filename(__name__, "models/shape_predictor_5_face_landmarks.dat")
        pose_predictor = dlib.shape_predictor(predictor_5_model_location)
        _raw_landmarks = [pose_predictor(image, face_location) for face_location in _raw_face_locations]

        # face recognition model v1 from dlib
        face_recognition_model_location = resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")
        face_encoder = dlib.face_recognition_model_v1(face_recognition_model_location)

        # compute_face_descriptor returns dlib.vectors; convert them to the numpy array
        return [np.array(face_encoder.compute_face_descriptor(image, raw_landmark, jitter))
                for raw_landmark in _raw_landmarks]

    @staticmethod
    def encoding_distance(known_encodings, encoding_check):
        if len(known_encodings) == 0:
            return np.empty(0)

        return np.linalg.norm(known_encodings - encoding_check, axis=1)

    @staticmethod
    def compare_encodings(known_encodings, encoding_check, tolerance=0.5):
        return list(FaceRecognition.encoding_distance(known_encodings, encoding_check) <= tolerance)


# webcam stream class

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# Face Recognition Process class

class FaceRecognitionProcess:
    def __init__(self, fx=0.0, fy=0.0, capture=None, known_encodings=[], known_names=[]):
        self.capture = capture
        self.stopped = False
        self.face_locations = []
        self.face_names = []
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.fx = fx
        self.fy = fy

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:

            # Grab a single frame from live video stream
            _frame = self.capture.read()

            if not (self.fx == 0.0 and self.fy == 0.0):
                # Resize the frame of video to 1/x size for faster face recognition processing
                _temp_frame = cv2.resize(_frame, (0, 0), fx=self.fx, fy=self.fy)

                # Convert the image from BGR color (opencv uses) to RGB color
                _frame = _temp_frame[:, :, ::-1]
            else:
                # Convert the image from BGR color (opencv uses) to RGB color
                _frame = _frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            _face_locations = FaceRecognition.face_locations(_frame)
            _face_names = []

            if len(self.known_encodings) > 0:
                # face detection and identification
                _face_encodings = FaceRecognition.face_encodings(_frame, locations=_face_locations)
                for face_encoding in _face_encodings:
                    # See if the face matches to known faces
                    matches = FaceRecognition.compare_encodings(known_encodings=self.known_encodings,
                                                                encoding_check=face_encoding)
                    _name = "Unknown"

                    # If found, use the first one
                    if True in matches:
                        first_match_index = matches.index(True)
                        _name = self.known_names[first_match_index]

                    _face_names.append(_name)

            else:
                # face detection only
                _face_names = ["person" for _ in _face_locations]

            self.face_locations = _face_locations
            self.face_names = _face_names

    def stop(self):
        self.stopped = True


# Main function

def main():

    # Replace with your images with its corresponding label
    # e.g., Assign John Doe to person1.jpg & Jane Roe to person2.jpg
    known_face_encodings = [
        FaceRecognition.face_encodings(FaceRecognition.load_image("person1.jpg"))[0],
        FaceRecognition.face_encodings(FaceRecognition.load_image("person2.jpg"))[0]
    ]

    known_face_names = [
        "John Doe",
        "Jane Roe"
    ]

    scale_factor = 0.5
    # e.g., r_scale_factor = 2 for scale_factor = 0.5
    r_scale_factor = int(1 / (scale_factor if scale_factor != 0.0 else 1))

    video_capture = WebcamVideoStream(src=0).start()

    # If you want to detect face without identifying from known faces, remove known_encodings & known_names parameters
    # when you initialize the class
    video_process = FaceRecognitionProcess(capture=video_capture,
                                           known_encodings=known_face_encodings,
                                           known_names=known_face_names,
                                           fx=scale_factor,
                                           fy=scale_factor).start()

    # Process the video rendering in Main thread due to opencv bug for Mac platform
    while True:

        if video_capture.stopped:
            video_capture.stop()
            break

        frame = video_capture.read()

        # Display the results
        locations = video_process.face_locations
        names = video_process.face_names

        for (left, top, right, bottom), name in zip(locations, names):
            # Scale up to the original size
            top *= r_scale_factor
            right *= r_scale_factor
            bottom *= r_scale_factor
            left *= r_scale_factor

            # Draw a box around the detected face  - BGR
            cv2.rectangle(frame, (left, top), (right, bottom), (244, 134, 66), 3)

            # Draw a label with a name
            cv2.rectangle(frame, (left-2, top - 35), (right+2, top), (244, 134, 66), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.stop()
    video_process.stop()

    cv2.destroyAllWindows()


# Call main() - entry point for the main program
main()
