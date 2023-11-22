function makePrediction() {
    const Age = parseInt(document.getElementById('Age').value);
    const RestingBP = parseInt(document.getElementById('RestingBP').value);
    const Cholesterol = parseInt(document.getElementById('Cholesterol').value);
    const FastingBS = parseInt(document.getElementById('FastingBS').value);
    const MaxHR = parseInt(document.getElementById('MaxHR').value);
    const Oldpeak = parseFloat(document.getElementById('Oldpeak').value);

    // Validate that Age is between 28 and 77
    if (Age < 28 || Age > 77) {
        alert('We are sorry, but this model was created with information based on people between 28 and 77 years old.');
        return;
    }

    // Validate that RestingBP is between 80 and 200
    if (RestingBP < 80 || RestingBP > 200) {
        alert('We are sorry, but this model was created with information based on people with Resting BP between 800 and 200.');
        return;
    }

    // Validate that Cholesterol is between 0 and 603
    if (Cholesterol < 0 || Cholesterol > 603) {
        alert('We are sorry, but this model was created with information based on people with Cholesterol between 0 and 603.');
        return;
    }

    // Validate that FastingBS is between 0 and 603
    if (FastingBS !== 0 && FastingBS !== 1) {
        alert('FastingBS must be either 0 or 1.');
        return;
    }

    // Validate that MaxHR is between 60 and 202
    if (MaxHR < 60 || MaxHR > 202) {
        alert('We are sorry, but this model was created with information based on people with MaxHR between 60 and 202.');
        return;
    }

    // Validate that Oldpeak is between -2.6 and 6.2
    if (Oldpeak < -2.6 || Oldpeak > 6.2) {
        alert('We are sorry, but this model was created with information based on people with Oldpeak between -2.6 and 6.2.');
        return;
    }    


    // Get the selected value from the category dropdown menu
    const selectSex = document.getElementById('Sex').value;
    const selectChestPainType = document.getElementById('ChestPainType').value;
    const selectedRestingECG = document.getElementById('RestingECG').value;
    const selectedExerciseAngina = document.getElementById('ExerciseAngina').value;
    const selectedST_Slope = document.getElementById('ST_Slope').value;

    const data = {
        Age,
        RestingBP,
        Cholesterol,
        MaxHR,
        Oldpeak,
        Sex_F: selectSex === 'F' ? 1 : 0,
        Sex_M: selectSex === 'M' ? 1 : 0,
        ChestPainType_ASY: selectChestPainType === 'ASY' ? 1 : 0,
        ChestPainType_ATA: selectChestPainType === 'ATA' ? 1 : 0,
        ChestPainType_NAP: selectChestPainType === 'NAP' ? 1 : 0,
        ChestPainType_TA: selectChestPainType === 'TA' ? 1 : 0,
        RestingECG_LVH: selectedRestingECG === 'LVH' ? 1 : 0,
        RestingECG_Normal: selectedRestingECG === 'Normal' ? 1 : 0,
        RestingECG_ST: selectedRestingECG === 'ST' ? 1 : 0,
        ExerciseAngina_N: selectedExerciseAngina === 'N' ? 1 : 0,
        ExerciseAngina_Y: selectedExerciseAngina === 'Y' ? 1 : 0,
        ST_Slope_Down: selectedST_Slope === 'Down' ? 1 : 0,
        ST_Slope_Flat: selectedST_Slope === 'Flat' ? 1 : 0,
        ST_Slope_Up: selectedST_Slope === 'Up' ? 1 : 0,
    };

    

    fetch('http://127.0.0.1:5000'/predict, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
    document.getElementById('predictionResult').innerText = `Prediction: ${result.prediction}`;
    })

  
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('predictionResult').innerText = 'Error making prediction';
    });
}