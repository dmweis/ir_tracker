# Ir tracker awful code

Scratch pad repo for ir beacon tracker.

This code is not interesting. This is only public so that I don't have to give keys to raspberry pis to push new code there.

## Dependencies

Depends on flask, opencv, numpy. Installation depends on what platform you are on.  
Generally I'd suggest using some virtual env system. But on a raspberry pi installing opencv with hardware acceleration in an environment can be.... finicky

### Install Dependencies on raspbian

```shell
apt install cmake libjpeg8-dev gcc g++ python3-dev python3-opencv
```

## Raspberry pi

Remember to enable the camera in `raspi-config`
