async function loadHITL(){

const response = await fetch("http://127.0.0.1:8000/hitl_queue")

const data = await response.json()

const table = document.getElementById("hitl-body")

table.innerHTML = ""

data.forEach(item => {

const row = document.createElement("tr")

row.innerHTML = `
<td>${item.question}</td>
<td>${item.answer}</td>
<td>${item.entropy}</td>
<td>${item.risk}</td>
<td>
<button class="correct">Correct</button>
<button class="incorrect">Incorrect</button>
</td>
`

table.appendChild(row)

const correctBtn = row.querySelector(".correct")
const incorrectBtn = row.querySelector(".incorrect")

correctBtn.onclick = () => {
    alert("Gunakan sistem pairwise di halaman utama (DPO)")
}

incorrectBtn.onclick = () => {
    alert("Gunakan sistem pairwise di halaman utama (DPO)")
}

})

}

async function evaluate(item,label){

await fetch("http://127.0.0.1:8000/evaluate",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body: JSON.stringify({
question: item.question,
answer: item.answer,
entropy: item.entropy,
label: label
})

})

alert("Evaluation Saved")

loadHITL()

}

loadHITL()