import face_recognition


class FaceRecognition:
    """
    Face recognition using the face_recognition library.
    """
    
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
    
    def add_known_face(self, image_path, name):
        """
        Add a known face from an image file.
        
        Args:
            image_path: Path to the image file.
            name: Name to associate with this face.
            
        Returns:
            bool: True if face was added successfully, False otherwise.
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                print(f"Warning: No face found in {image_path}")
                return False
            
            encoding = encodings[0]
            self.known_encodings.append(encoding)
            self.known_names.append(name)
            return True
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False
    
    def recognize(self, frame):
        """
        Recognize faces in a frame.
        """
        try:
            rgb_frame = frame[:, :, ::-1]  # BGR → RGB
            locations = face_recognition.face_locations(rgb_frame, model="hog")
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            names = []
            for encoding in encodings:
                if len(self.known_encodings) == 0:
                    names.append("Unknown")
                    continue

                face_distances = face_recognition.face_distance(self.known_encodings, encoding)
                name = "Unknown"

                if len(face_distances) > 0:
                    best_match_index = face_distances.argmin()
                    if face_distances[best_match_index] < 0.5:
                        name = self.known_names[best_match_index]

                names.append(name)

            return locations, names

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return [], []

    def clear_known_faces(self):
        """Clear all known faces."""
        self.known_encodings = []
        self.known_names = []
    
    def get_known_face_count(self):
        """Get the number of registered known faces."""
        return len(self.known_names)

