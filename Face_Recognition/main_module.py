import face_recognition as face
import cv2
import os

# Tolerance defines the face matching level (Recommended 0.6)
# Higher tolerance > higher the correct matching
# Lower tolerance > more false positives
match_tolerance = 0.6

# Set up input data directory path
base_dir_path = '/Users/test/Documents/git_workspace/face_recognition'
known_faces_dir_path = os.path.join(base_dir_path, 'known_faces')
unknown_faces_dir_path = os.path.join(base_dir_path, 'unknown_faces')

# Parameters for the identification frame around the face
frame_thickness = 3
font_thickness = 2

# Model being used (CNN, HOG(older and faster for running on CPU), etc.)
face_detection_model = 'cnn'

# Face Recognition steps:
# 1. Identify faces
# 2. Match those faces from previously known faces
known_faces_list = []
known_names_list = []

for person_name in os.listdir(known_faces_dir_path):
    for file_name in os.listdir(os.path.join(known_faces_dir_path, person_name)):
        image = face.load_image_file(os.path.join(
            known_faces_dir_path,
            person_name,
            file_name
        ))

        print(file_name)
        # 0th index so it encodes only the first face
        # Also, these images should contain only the identified person's image
        image_encoding = face.face_encodings(image)[0]
        
        # Add these entries in the image lists
        known_faces_list.append(image_encoding)
        known_names_list.append(person_name)

print('Known image scanning complete')


# Identify faces in unknown images
for file_name in os.listdir(unknown_faces_dir_path):
    print('Found new file name: ', file_name)

    # Load the image
    image = face.load_image_file(os.path.join(unknown_faces_dir_path, file_name))

    # Detect locations of all the faces present in the image
    image_face_locations = face.face_locations(image, model=face_detection_model)

    # Encode all the faces found at the face locations
    image_encoding = face.face_encodings(image, image_face_locations)

    # Convert image to be usable with opencv
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Iterate over all unknown faces & their encodings and match them with
    # known face encodings
    for unknown_encoding, unknown_location in zip(image_encoding, image_face_locations):
        comparison_results = face.compare_faces(
            known_faces_list,
            unknown_encoding,
            tolerance=match_tolerance
        )
        # results contains boolean value for matched and unmatched faces

        matched_name = None
        if True in comparison_results:
            matched_name = known_names_list[comparison_results.index(True)]
            print('Matches found: ', matched_name)

            # Drawing boxes around matched faces using cv2
            top_left_coordinate = (unknown_location[3], unknown_location[0])
            bottom_right_coordinate = (unknown_location[1], unknown_location[2])

            # We can also have dynamic color box dependent on matches
            box_color = [0, 255, 0]

            cv2.rectangle(
                image,
                top_left_coordinate,
                bottom_right_coordinate,
                box_color,
                thickness=frame_thickness
            )

            # Another box for displaying the name
            top_left_coordinate = (unknown_location[3], unknown_location[2])
            bottom_right_coordinate = (unknown_location[1], unknown_location[2]+22)
            cv2.rectangle(
                image,
                top_left_coordinate,
                bottom_right_coordinate,
                box_color,
                cv2.FILLED
            )
            cv2.putText(
                image,
                matched_name,
                (unknown_location[3]+15, unknown_location[2]+15),
                fontFace=cv2.FONT_ITALIC,
                fontScale=0.5,
                color=(200, 200, 200),
                thickness=font_thickness,
            )
    print('Tagging complete')
    cv2.imshow(file_name, image)
    cv2.waitKey(1000000)
    # cv2.destroyWindow(file_name)


