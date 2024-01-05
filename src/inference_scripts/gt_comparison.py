from codec import encoding

def name_checker(name):
    """
    Checks if the given name is in a predefined list of names.

    Parameters:
    - name (str): The name to be checked.

    Returns:
    - str: The corrected or validated name.

    If the provided name is not in the predefined list, the function prompts the user
    to enter a correct name from a predefined list and recursively calls itself until a
    valid name is provided.
    """
    names = ['flag', 'fl', 'bl', 'oafl', 'oahs', 'pl', 'ic', 'mal', 'vsit']

    # Remove leading and trailing whitespaces
    if name.startswith(" "):
        name = name[1:]
    if name.endswith(" "):
        name = name[:-1]

    # Check if the name is in the predefined list
    if name not in names:
        print("The name you inserted is not correct, please insert the correct name")
        print("Choose one between these : {'flag', 'fl', 'bl', 'oafl', 'oahs', 'pl', 'ic', 'mal', 'vsit'}")
        name = input()
        # Recursively call the function with the new input until a valid name is provided
        name = name_checker(name)

    return name



def gtc_algorithm(seconds_length, total_frames):
    """
    Perform the GTC (Ground Truth Comparison) algorithm.

    Parameters:
    - seconds_length (int): The length of the video in seconds.
    - total_frames (int): The total number of frames in the video.

    Returns:
    - tuple: A tuple containing:
        - list: The ground truth labels encoded.
        - list: The list of skill frames.
        - list: The list of skill durations in seconds.
    """
    print("How many skills are there in the video?")
    try: 
        n_skills = int(input())
    except ValueError:
        print("Please insert a number")
        n_skills = int(input())

    print("There are", n_skills, "skills in the video")

    skills_frames = []
    skills_seconds = []

    print("Names : {'flag', 'fl', 'bl', 'oafl', 'oahs', 'pl', 'ic', 'mal', 'vsit'}")
    for i_skill in range(0, n_skills):
        print("Insert the name of the #", i_skill+1,"skill in the video")
        skill_name = input()
        skill_name = name_checker(skill_name)
        print("Insert the start frame of the #", i_skill+1,"skill in the video")
        try:
            start_frame = int(input())
        except ValueError:
            print("Please insert a number")
            start_frame = int(input())
        if start_frame < 0:
            start_frame = 0

        print("Insert the end frame of the #", i_skill+1,"skill in the video")
        try:
            end_frame = int(input())
        except ValueError:
            print("Please insert a number")
            end_frame = int(input())

        if end_frame > total_frames:
            end_frame = total_frames-1
        skills_frames.append([skill_name, start_frame, end_frame])

    # Reconstruct none sequence
    i = 0
    while True:
        if i == 0 and skills_frames[i][1] != 0:
            skills_frames.insert(0, ["none", 0, skills_frames[i][1]-1])
            seconds_length += 1
            
        if i != len(skills_frames)-1:
            if skills_frames[i][2]+1 != skills_frames[i+1][1]:
                skills_frames.insert(i+1, ["none", skills_frames[i][2]+1, skills_frames[i+1][1]-1])
                seconds_length += 1
            
        if i == len(skills_frames)-1:
            if skills_frames[i][2]+1 < total_frames-1:
                skills_frames.append(["none", skills_frames[i][2]+1, total_frames-1]) 
            break

        i = i+1

    # Converting the frames in seconds into a new list
    for i in range(0, len(skills_frames)):
        skills_seconds.append([skills_frames[i][0], (skills_frames[i][2]-skills_frames[i][1])/24])

    labels_predicted = []
    for i in range(0, len(skills_frames)):
        for j in range(skills_frames[i][1], skills_frames[i][2]+1):
            labels_predicted.append(skills_frames[i][0])

    labels_predicted = encoding(labels_predicted)

    print("Labels predicted", labels_predicted)
    print("Skills in frames: ", skills_frames)
    print("Skills in seconds: ", skills_seconds)

    return labels_predicted, skills_frames, skills_seconds
