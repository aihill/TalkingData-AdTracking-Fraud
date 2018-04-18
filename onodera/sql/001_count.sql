/*
Created on Sat Mar 17 09:54:47 2018

@author: Kazuki
*/

-- save as ip_count
SELECT ip, COUNT(*) as ip_count
FROM TalkingData.train, TalkingData.test_old
GROUP BY ip

-- save as app_count
SELECT app, COUNT(*) as app_count
FROM TalkingData.train, TalkingData.test_old
GROUP BY app

-- save as device_count
SELECT device, COUNT(*) as device_count
FROM TalkingData.train, TalkingData.test_old
GROUP BY device

-- save as os_count
SELECT os, COUNT(*) as os_count
FROM TalkingData.train, TalkingData.test_old
GROUP BY os

-- save as channel_count
SELECT channel, COUNT(*) as channel_count
FROM TalkingData.train, TalkingData.test_old
GROUP BY channel


