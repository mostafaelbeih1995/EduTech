import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Platform } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from '@tensorflow-models/mobilenet';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import WebView from 'react-native-webview';
import * as ScreenOrientation from 'expo-screen-orientation';
import { TikTok } from 'react-tiktok';

const textureDims = Platform.OS === 'ios' ?
  {
    height: 1920,
    width: 1080,
  } :
   {
    height: 1200,
    width: 1600,
  };

let frame = 0;
const computeRecognitionEveryNFrames = 60;

const TensorCamera = cameraWithTensors(Camera);

const initialiseTensorflow = async () => {
  await tf.ready();
  tf.getBackend();
}



export default function App() {
  const [hasPermission, setHasPermission] = useState<null | boolean>(null);
  const [detections, setDetections] = useState<string[]>([]);
  const [net, setNet] = useState<mobilenet.MobileNet>();


  const handleCameraStream = (images: IterableIterator<tf.Tensor3D>) => {
    const loop = async () => {
      if(net) {
        if(frame % computeRecognitionEveryNFrames === 0){
          const nextImageTensor = images.next().value;
          if(nextImageTensor){
            const objects = await net.classify(nextImageTensor);
            if(objects && objects.length > 0){
              setDetections(objects.map((object: { className: any; }) => object.className));
            }
            tf.dispose([nextImageTensor]);
          }
        }
        frame += 1;
        frame = frame % computeRecognitionEveryNFrames;
      }

      requestAnimationFrame(loop);
    }
    loop();
  }

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
      await initialiseTensorflow();
      setNet(await mobilenet.load({ version: 1, alpha: 0.25 }));

      //for camera rotation
      // await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.LANDSCAPE_LEFT);
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }
  if(!net){
    return <Text>Model not loaded</Text>;
  }

  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        onReady={handleCameraStream}
        type={Camera.Constants.Type.back}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        autorender={true} useCustomShadersToResize={false}/>
      <View style={styles.text}>
      {detections.map((detection, index) =>
          <Text key={index}>{detection}</Text>
      )}
      </View>
    </View>

    //Simulation lesson view 


    //@ts-ignore
    // <View style={styles.container}>
    //   <WebView
    //     source={{html: '<iframe src="https://phet.colorado.edu/sims/html/density/latest/density_en.html" width="100%" height="100%" allowFullScreen: {true}></iframe></WebView>'}}
    //     style={styles.container}
    // />
    // </View>


    //Embeding tiktok Video View
    // <>
    //   <blockquote className="tiktok-embed" cite="https://www.tiktok.com/@quttaii/video/7069006975203347713" data-video-id="7069006975203347713">
    //     <section>
    //       <a target="_blank" title="@quttaii" href="https://www.tiktok.com/@quttaii">@quttaii</a>
    //       <a target="_blank" title="♬ оригинальный звук - aida_serikova_2000" href="https://www.tiktok.com/music/оригинальный-звук-7069007007663147777">♬ оригинальный звук - aida_serikova_2000</a>
    //     </section>
    //   </blockquote>
    //   <script async src="https://www.tiktok.com/embed.js"></script>
    // </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    flex: 1,
  },
  camera: {
    flex: 10,
    width: '100%',
  },
});



