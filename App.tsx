import React, { useEffect, useState } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  Image,
  Button,
} from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

function App(): React.JSX.Element {
  const [isTfReady, setTfReady] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [detections, setDetections] = useState<string[]>([]);

  useEffect(() => {
    // Inicializar TensorFlow y cargar el modelo
    const initializeTensorFlow = async () => {
      await tf.ready(); // Asegura que TensorFlow esté listo
      setTfReady(true);
      const loadedModel = await cocoSsd.load(); // Cargar modelo Coco SSD
      setModel(loadedModel);
    };
    initializeTensorFlow();
  }, []);

  const handleDetectObjects = async () => {
    if (!model) return;
    const imageTensor = tf.browser.fromPixels({
      width: 640,
      height: 480,
      data: new Uint8Array(640 * 480 * 3), // Debe reemplazarse con la imagen real
    });
    
    const predictions = await model.detect(imageTensor);
    setDetections(predictions.map((prediction: any) => prediction.class));
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentInsetAdjustmentBehavior="automatic">
        <View style={styles.content}>
          <Text style={styles.title}>Demo: TensorFlow Coco SSD</Text>
          {isTfReady ? (
            <Text>TensorFlow está listo</Text>
          ) : (
            <Text>Cargando TensorFlow...</Text>
          )}
          <Button
            title="Detectar objetos"
            onPress={handleDetectObjects}
            disabled={!model}
          />
          <View style={styles.detectionsContainer}>
            {detections.map((item, index) => (
              <Text key={index} style={styles.detectionText}>
                {item}
              </Text>
            ))}
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  content: {
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  detectionsContainer: {
    marginTop: 20,
  },
  detectionText: {
    fontSize: 18,
    marginVertical: 5,
  },
});

export default App;
