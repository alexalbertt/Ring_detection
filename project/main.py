import ring_video
import object_detection
import yagmail
import account

# get video of last motion alert
video_name = ring_video.doorbellCamVideo()

# identify the moving object and create annotated video
moving_objects = object_detection.run_script(video_name)

# get the time the motion alert occurred
time_detected = ring_video.doorbellCamMotion()

# only have to run this line once to save gmail account details in keyring
yagmail.register(account.email, account.password)

# create an SMTP connection
yag = yagmail.SMTP(account.email)

# send the email to your email
subject = "Ring Motion Detection"
contents = [
    f"A {moving_objects[0]} was detected moving at {time_detected}.",
    "detected_video.avi",
]
yag.send(account.email, subject, contents)
