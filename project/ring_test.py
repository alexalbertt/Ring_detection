from ring_doorbell import Ring
import account

# set up Ring object with account details
myring = Ring(account.email, account.password)
video_name = "last_motion.mp4"
ringDoorbells = myring.doorbells

def CameraCheck():
    for dev in list(ringDoorbells):

        dev.update()

        print('Account ID: %s' % dev.account_id)
        print('Device Type: %s' % dev.family)
        print('Address:    %s' % dev.address)
        print('Family:     %s' % dev.family)
        print('ID:         %s' % dev.id)
        print('Name:       %s' % dev.name)
        print('Timezone:   %s' % dev.timezone)
        print('Wifi Name:  %s' % dev.wifi_name)
        print('Wifi RSSI:  %s' % dev.wifi_signal_strength)
        print('Battery Life: %s' % dev.battery_life)

def doorbellCamVideo(video_name):
        doorbell = myring.doorbells[0]
        doorbell.recording_download(
            doorbell.history(limit=100, kind='motion')[0]['id'],
                                    filename=video_name, override=True)
        return video_name

def doorbellCamMotion():
        for doorbell in ringDoorbells:
            #You can change the limit
            for event in doorbell.history(limit=1):
                print('Footage ID:     %s' % event['id'])
                print('Kind:     %s' % event['kind'])
                print('Answered:     %s' % event['answered'])
                print('Date/Time:     %s' % event['created_at'])
                return event['created_at']

if __name__ == "__main__":
    CameraCheck()
    doorbellCamVideo(video_name)
    doorbellCamMotion()
