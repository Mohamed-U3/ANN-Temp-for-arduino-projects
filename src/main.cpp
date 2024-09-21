#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT22 // or DHT11
DHT dht(DHTPIN, DHTTYPE);

#define LEDCold     12
#define LEDComfort  14
#define LEDHot      27

/******************************************************************
   ArduinoANN - An artificial neural network for the Arduino
   All basic settings can be controlled via the Network Configuration
   section.
   See robotics.hobbizine.com/arduinoann.html for details.
 ******************************************************************/

#include <math.h>

/******************************************************************
   Network Configuration - customized per network
 ******************************************************************/

const int PatternCount = 6;
const int InputNodes = 2;
const int HiddenNodes = 6;
const int OutputNodes = 3;
const float LearningRate = 0.3;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.0004;

/* The input data - represents an input pattern */
const float Input[PatternCount][InputNodes] = {
  {0.2, 0.3}, // Cold/Dry
  {0.3, 0.4}, // Cold/Dry
  {0.5, 0.5}, // Comfortable
  {0.6, 0.6}, // Comfortable
  {0.8, 0.7}, // Hot/Humid
  {0.9, 0.8}  // Hot/Humid
};

/* The target data - represents the desired output for each input pattern. */
const byte Target[PatternCount][OutputNodes] = {
  {1, 0, 0}, // Cold
  {1, 0, 0}, // Cold
  {0, 1, 0}, // Comfortable
  {0, 1, 0}, // Comfortable
  {0, 0, 1}, // Hot
  {0, 0, 1}  // Hot
};

/******************************************************************
   End Network Configuration
 ******************************************************************/



/******************************************************************
 * Functions declireation 
 ******************************************************************/
void toTerminal();
void ANNTrain();
void ANNInference(const float newInput[InputNodes], float predictedOutput[OutputNodes]);

/******************************************************************
 * Variables declireation 
 ******************************************************************/
int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error;
float Accum;


float Hidden[HiddenNodes];                               /* store the activations of the hidden layers*/
float Output[OutputNodes];                               /* store the activations of the output layers*/
float HiddenWeights[InputNodes + 1][HiddenNodes];        /* store the weights between input-hidden layers*/
float OutputWeights[HiddenNodes + 1][OutputNodes];       /* store the weights between hidden-output layers*/
float HiddenDelta[HiddenNodes];                          /* store the error signals for the hidden layers during backpropagation*/
float OutputDelta[OutputNodes];                          /* store the error signals for the output layers during backpropagation*/
float ChangeHiddenWeights[InputNodes + 1][HiddenNodes];  /* store the change in weights from the previous iteration*/
float ChangeOutputWeights[HiddenNodes + 1][OutputNodes]; /* store the change in weights from the previous iteration*/

void setup()
{
  Serial.begin(9600);
  dht.begin();
  pinMode(LEDCold,    OUTPUT);
  pinMode(LEDComfort, OUTPUT);
  pinMode(LEDHot,     OUTPUT);

  randomSeed(analogRead(3));
  ReportEvery1000 = 1;
  for ( p = 0 ; p < PatternCount ; p++ )
  {
    RandomizedIndex[p] = p ;
  }
  ANNTrain();
}

void loop()
{
  // Read sensor values
  float temp = dht.readTemperature(); // in Celsius
  float hum = dht.readHumidity();     // in %

  if (isnan(temp) || isnan(hum))
  {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Normalize inputs (assuming temp range 0-50°C and hum 0-100%)
  float normTemp = temp / 50.0;
  float normHum = hum / 100.0;

  // Define a new input pattern you want to test
  const float newInput[InputNodes] = {normTemp, normHum}; // Example new input
  float predictedOutput[OutputNodes];
  uint8_t predictedOutputIO[OutputNodes];

  // Run inference with the trained weights
  ANNInference(newInput, predictedOutput);
  // Output the result as 1 or 0 only
  Serial.println("Prediction for new input 0-1:");
  for (int i = 0; i < OutputNodes; i++)
  {
    predictedOutputIO[i] = round(predictedOutput[i]);
    Serial.print(predictedOutputIO[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Determine the category with the highest output value
  int predictedCategory = 0;
  float maxVal = predictedOutput[0];
  for (int i = 1; i < OutputNodes; i++)
  {
    if (predictedOutput[i] > maxVal) {
      maxVal = predictedOutput[i];
      predictedCategory = i;
    }
  }

  // Control LEDs based on classification
  digitalWrite(LEDCold,     predictedCategory == 0 ? HIGH : LOW);
  digitalWrite(LEDComfort,  predictedCategory == 1 ? HIGH : LOW);
  digitalWrite(LEDHot,      predictedCategory == 2 ? HIGH : LOW);

  // Optional: Print classification to Serial Monitor
  Serial.print("Temp: "); Serial.print(temp);
  Serial.print("°C, Humidity: "); Serial.print(hum);
  Serial.print("% => ");
  if (predictedCategory == 0)       Serial.println("Cold/Dry");
  else if (predictedCategory == 1)  Serial.println("Comfortable");
  else if (predictedCategory == 2)  Serial.println("Hot/Humid");

  // Add a delay for demonstration purposes
  delay(5000);
}


/******************************************************************
 * Functions Definition
 ******************************************************************/
void ANNTrain()
{
  /******************************************************************
    Initialize HiddenWeights and ChangeHiddenWeights
  ******************************************************************/

  for ( i = 0 ; i < HiddenNodes ; i++ )
  {
    for ( j = 0 ; j <= InputNodes ; j++ )
    {
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100)) / 100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
  /******************************************************************
    Initialize OutputWeights and ChangeOutputWeights
  ******************************************************************/

  for ( i = 0 ; i < OutputNodes ; i ++ )
  {
    for ( j = 0 ; j <= HiddenNodes ; j++ )
    {
      ChangeOutputWeights[j][i] = 0.0 ;
      Rando = float(random(100)) / 100;
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
  Serial.println("Initial/Untrained Outputs: ");
  toTerminal();
  /******************************************************************
    Begin training
  ******************************************************************/

  for ( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++)
  {

    /******************************************************************
      Randomize order of training patterns
    ******************************************************************/

    for ( p = 0 ; p < PatternCount ; p++)
    {
      q = random(PatternCount);
      r = RandomizedIndex[p] ;
      RandomizedIndex[p] = RandomizedIndex[q] ;
      RandomizedIndex[q] = r ;
    }
    Error = 0.0 ;
    /******************************************************************
      Cycle through each training pattern in the randomized order
    ******************************************************************/
    for ( q = 0 ; q < PatternCount ; q++ )
    {
      p = RandomizedIndex[q];

      /******************************************************************
        Compute hidden layer activations
      ******************************************************************/

      for ( i = 0 ; i < HiddenNodes ; i++ )
      {
        Accum = HiddenWeights[InputNodes][i] ;
        for ( j = 0 ; j < InputNodes ; j++ )
        {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;
      }

      /******************************************************************
        Compute output layer activations and calculate errors
      ******************************************************************/

      for ( i = 0 ; i < OutputNodes ; i++ )
      {
        Accum = OutputWeights[HiddenNodes][i];
        for ( j = 0 ; j < HiddenNodes ; j++ )
        {
          Accum += Hidden[j] * OutputWeights[j][i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]);
      }

      /******************************************************************
        Backpropagate errors to hidden layer
      ******************************************************************/

      for ( i = 0 ; i < HiddenNodes ; i++ )
      {
        Accum = 0.0 ;
        for ( j = 0 ; j < OutputNodes ; j++ )
        {
          Accum += OutputWeights[i][j] * OutputDelta[j];
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]);
      }


      /******************************************************************
        Update Inner-->Hidden Weights
      ******************************************************************/


      for ( i = 0 ; i < HiddenNodes ; i++ )
      {
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for ( j = 0 ; j < InputNodes ; j++ )
        {
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

      /******************************************************************
        Update Hidden-->Output Weights
      ******************************************************************/

      for ( i = 0 ; i < OutputNodes ; i ++ )
      {
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for ( j = 0 ; j < HiddenNodes ; j++ )
        {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

    /******************************************************************
      Every 1000 cycles send data to terminal for display
    ******************************************************************/
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0)
    {
      Serial.println();
      Serial.println();
      Serial.print ("TrainingCycle: ");
      Serial.print (TrainingCycle);
      Serial.print ("  Error = ");
      Serial.println (Error, 5);

      toTerminal();

      if (TrainingCycle == 1)
      {
        ReportEvery1000 = 999;
      }
      else
      {
        ReportEvery1000 = 1000;
      }
    }


    /******************************************************************
      If error rate is less than pre-determined threshold then end
    ******************************************************************/

    if ( Error < Success ) break ;
  }
  Serial.println ();
  Serial.println();
  Serial.print ("TrainingCycle: ");
  Serial.print (TrainingCycle);
  Serial.print ("  Error = ");
  Serial.println (Error, 5);

  toTerminal();

  Serial.println ();
  Serial.println ();
  Serial.println ("Training Set Solved! ");
  Serial.println ("--------");
  Serial.println ();
  Serial.println ();
  ReportEvery1000 = 1;

}

void toTerminal()
{
  for ( p = 0 ; p < PatternCount ; p++ )
  {
    Serial.println();
    Serial.print ("  Training Pattern: ");
    Serial.println (p);
    Serial.print ("  Input ");
    for ( i = 0 ; i < InputNodes ; i++ )
    {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Target ");
    for ( i = 0 ; i < OutputNodes ; i++ )
    {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }
    /******************************************************************
      Compute hidden layer activations
    ******************************************************************/

    for ( i = 0 ; i < HiddenNodes ; i++ )
    {
      Accum = HiddenWeights[InputNodes][i] ;
      for ( j = 0 ; j < InputNodes ; j++ )
      {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

    /******************************************************************
      Compute output layer activations and calculate errors
    ******************************************************************/

    for ( i = 0 ; i < OutputNodes ; i++ )
    {
      Accum = OutputWeights[HiddenNodes][i] ;
      for ( j = 0 ; j < HiddenNodes ; j++ )
      {
        Accum += Hidden[j] * OutputWeights[j][i];
      }
      Output[i] = 1.0 / (1.0 + exp(-Accum));
    }
    Serial.print ("  Output ");
    for ( i = 0 ; i < OutputNodes ; i++ )
    {
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
  }
}

void ANNInference(const float newInput[InputNodes], float predictedOutput[OutputNodes])
{
  float Hidden[HiddenNodes];
  float Output[OutputNodes];
  float Accum;

  // Calculate activations for the hidden layer
  for (int i = 0; i < HiddenNodes; i++)
  {
    Accum = HiddenWeights[InputNodes][i]; // Bias term
    for (int j = 0; j < InputNodes; j++)
    {
      Accum += newInput[j] * HiddenWeights[j][i]; // Input * weight
    }
    Hidden[i] = 1.0 / (1.0 + exp(-Accum)); // Sigmoid activation
  }

  // Calculate activations for the output layer
  for (int i = 0; i < OutputNodes; i++)
  {
    Accum = OutputWeights[HiddenNodes][i]; // Bias term
    for (int j = 0; j < HiddenNodes; j++)
    {
      Accum += Hidden[j] * OutputWeights[j][i]; // Hidden * weight
    }
    predictedOutput[i] = 1.0 / (1.0 + exp(-Accum)); // Sigmoid activation
  }

  // Output the result
  Serial.println("Prediction for new input:");
  for (int i = 0; i < OutputNodes; i++)
  {
    Serial.print(predictedOutput[i], 5);
    Serial.print(" ");
  }
  Serial.println();
}


