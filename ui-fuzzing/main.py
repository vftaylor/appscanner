import subprocess
import re
import time

NUMBER_OF_LAUNCHES = 0 # how many times to launch app

packages = ['com.instagram.android', 'com.snapchat.android', 'com.spotify.music', 'com.skype.raider',
            'air.uk.co.bbc.android.mediaplayer', 'com.lazyswipe', 'com.ebay.mobile',
            'com.prettysimple.criminalcaseandroid', 'com.netflix.mediaclient', 'bbc.mobile.weather',
            'com.instagram.layout', 'com.bigkraken.thelastwar', 'com.hcg.cok.gp', 'com.miniclip.eightballpool',
            'com.amazon.mShop.android.shopping', 'com.chatous.pointblank', 'bbc.iplayer.android',
            'com.jellybtn.cashkingmobile', 'com.twitter.android', 'com.gameloft.android.ANMP.GloftDMHM',
            'com.king.candycrushsaga', 'com.yodo1.crossyroad', 'com.nordcurrent.canteenhd',
            'com.king.candycrushsodasaga', 'com.supercell.clashofclans', 'com.playfirst.cookingdashx',
            'com.surpax.ledflashlight.panel', 'com.leftover.CoinDozer', 'com.boombit.Spider', 'com.pinterest',
            'com.contextlogic.wish', 'com.kiloo.subwaysurf', 'com.soundcloud.android', 'com.boombit.RunningCircles',
            'com.igg.android.finalfable', 'air.ITVMobilePlayer', 'com.shpock.android', 'net.zedge.android',
            'com.machinezone.gow', 'com.cleanmaster.mguard', 'com.whatsapp', 'com.facebook.orca',
            'com.king.alphabettysaga', 'com.google.android.apps.photos', 'com.tinder', 'com.facebook.katana',
            'com.microsoft.office.outlook', 'com.ciegames.RacingRivals', 'com.outplayentertainment.bubbleblaze',
            'com.gumtree.android', 'com.shazam.android', 'kik.android', 'com.imangi.templerun2',
            'com.google.android.apps.plus', 'com.rovio.angrybirds', 'com.google.android.youtube',
            'com.google.android.apps.maps', 'com.google.android.gm', 'com.halfbrick.fruitninjafree',
            'com.king.farmheroessaga','com.viber.voip', 'com.avast.android.mobilesecurity', 'com.qihoo.security',
            'com.mobilityware.solitaire', 'com.google.earth', 'com.yahoo.mobile.client.android.mail',
            'com.amazon.kindle', 'com.cnn.mobile.android.phone', 'com.supercell.boombeach', 'com.bskyb.skygo',
            'com.dropbox.android', 'com.outfit7.mytalkingtomfree', 'me.pou.app', 'com.mobilemotion.dubsmash',
            'com.google.android.apps.inbox', 'com.badoo.mobile', 'com.tripadvisor.tripadvisor',
            'com.justeat.app.uk', 'com.zentertain.photoeditor', 'com.mixradio.droid',
            'com.digidust.elokence.akinator.freemium', 'com.fivestargames.slots', 'com.myfitnesspal.android',
            'com.umonistudio.tile', 'com.groupon', 'uk.co.dominos.android', 'com.tayu.tau.pedometer',
            'hu.tonuzaba.android', 'tv.twitch.android.app', 'com.thetrainline',
            'air.com.puffballsunited.escapingtheprison', 'com.itv.loveislandapp', 'com.adobe.air',
            'com.robtopx.geometryjumplite', 'com.imo.android.imoim', 'com.BitofGame.MiniGolfRetro',
            'com.fungames.flightpilot', 'com.miniclip.agar.io', 'com.joycity.warshipbattle',
            'uk.co.nationalrail.google', 'com.meetup', 'info.androidz.horoscope', 'com.dictionary',
            'com.northpark.drinkwater', 'com.runtastic.android', 'com.imdb.mobile', 'com.joelapenna.foursquared',
            'com.booking', 'com.airbnb.android', 'com.iconology.comics']

#packages = packages[:10]

print len(packages)

for package in packages:
    p = subprocess.Popen('monkeyrunner.bat monkey-handler.py'
                     + package + ' ' + str(NUMBER_OF_LAUNCHES), stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        print line,
    p.communicate() # close p.stdout, wait for the subprocess to exit
    print 'FINISHED ALL RUNS FOR %s' % package
    time.sleep(1)
