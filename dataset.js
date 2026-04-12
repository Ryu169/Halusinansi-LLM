async function loadDataset(){

let response = await fetch("http://127.0.0.1:8000/dataset")

let data = await response.json()

let table = document.getElementById("table")

data.forEach(item => {

let row = table.insertRow()

row.insertCell(0).innerText = item.question
row.insertCell(1).innerText = item.entropy
row.insertCell(2).innerText = item.risk
row.insertCell(3).innerText = item.timestamp

})

}

loadDataset()


/* =============================
DATASET EXPORT FUNCTIONS
============================= */

function exportQA(){

window.open("http://127.0.0.1:8000/dataset")

}

function exportPreferences(){

window.open("http://127.0.0.1:8000/preferences")

}

function exportDPO(){

window.open("http://127.0.0.1:8000/export_dpo")

}

function exportDPOFile(){

window.open("http://127.0.0.1:8000/export_dpo_file")

}