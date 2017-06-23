import sys
from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice

package = sys.argv[1]
NUMBER_OF_LAUNCHES = int(sys.argv[2])
WAIT_TIME_BETWEEN_LAUNCHES = 4

print "CONNECTING to device."
device = MonkeyRunner.waitForConnection()

apk_path = device.shell('pm path ' + package)
if apk_path.startswith('package:'):
    print "Package %s is already installed." % package
else:
    print "Package %s is NOT installed, installing now." % package
    device.installPackage('apps\\' + package + '.apk')
    print "Package %s successfully installed." % package

for i in range(NUMBER_OF_LAUNCHES):

    # ensure airplane mode is off. sometimes the monkey turns it on
    device.shell('settings put global airplane_mode_on 0')
    device.shell('am broadcast -a android.intent.action.AIRPLANE_MODE --ez state false')

    print "Iteration number %s of %s" % ((i+1), NUMBER_OF_LAUNCHES)
    print "    Launching package %s" % package

    device.shell('monkey -p %s -c android.intent.category.LAUNCHER 1' % package)
    MonkeyRunner.sleep(4)
    print "    Running app with Monkey events..."
    # this is approximately 1 minute per run -> monkey --pct-syskeys 0 --pct-majornav 0 --pct-nav 0 --pct-trackball 0 --pct-motion 0 --pct-anyevent 0 --pct-appswitch 20 --throttle 500 -vv -p %s 240
    device.shell('monkey --pct-syskeys 0 --pct-majornav 0 --pct-nav 0 --pct-trackball 0 --pct-motion 0 --pct-anyevent 0 --pct-appswitch 20 --throttle 500 -vv -p %s 240' % package)

    print "    Terminating process %s" % package
    device.press('KEYCODE_HOME', MonkeyDevice.DOWN_AND_UP)
    device.shell('am force-stop %s' % package)
    
    if (i != (NUMBER_OF_LAUNCHES - 1)): # no need to wait between launches if it's the last iteration
        print "    Waiting for %s seconds..." % WAIT_TIME_BETWEEN_LAUNCHES
        MonkeyRunner.sleep(WAIT_TIME_BETWEEN_LAUNCHES)

if (NUMBER_OF_LAUNCHES != 0): # sometimes we set number of launches to 0 just to install the app and nothing else
    #### LAUNCH A URL IN THE WEB BROWSER (THEN WAIT 10 SECONDS) TO MARK THE END OF AN APP RUN
    MonkeyRunner.sleep(10)
    url = 'http://www.google.co.uk/search?' + package + '+uifuzzingruncomplete'
    device.shell('am start -a android.intent.action.VIEW -d %s' % url)
    MonkeyRunner.sleep(10)
    device.press('KEYCODE_HOME', MonkeyDevice.DOWN_AND_UP)
    device.shell('am force-stop com.android.chrome')
    print "Marked end of %s run with HTTP REQUEST" % package

#### UNINSTALL PACKAGE TO SAVE SPACE
# device.shell('pm uninstall ' + package)
# print "Package successfully uninstalled."

print "Done!"
