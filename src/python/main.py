import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    while True:
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(image_rgb)
        results_face = face_mesh.process(image_rgb)

        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            landmarks = results_pose.pose_landmarks.landmark
            if landmarks[16].y < landmarks[12].y:
                print("Mão direita levantada")
            if landmarks[15].y < landmarks[11].y:
                print("Mão esquerda levantada")

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Olho direito: ponto 159 (superior), 145 (inferior)
                right_eye_top = face_landmarks.landmark[159]
                right_eye_bottom = face_landmarks.landmark[145]

                # Olho esquerdo: ponto 386 (superior), 374 (inferior)
                left_eye_top = face_landmarks.landmark[386]
                left_eye_bottom = face_landmarks.landmark[374]

                # Calcular distância vertical dos olhos
                right_eye_diff = abs(right_eye_top.y - right_eye_bottom.y)
                left_eye_diff = abs(left_eye_top.y - left_eye_bottom.y)

                # Heurística simples: se a distância for muito pequena, o olho está fechado
                if right_eye_diff < 0.01:
                    print("Olho direito fechado")
                if left_eye_diff < 0.01:
                    print("Olho esquerdo fechado")

        cv2.imshow("MediaPipe Pose + FaceMesh", cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
