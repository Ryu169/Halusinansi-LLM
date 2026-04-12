let isLoading = false

async function askQuestion(){

    if (isLoading) return
    isLoading = true

    try {

        let question = document.getElementById("question").value

        // 🔥 tampilkan loading dulu
        document.getElementById("answer").innerText = "Generating..."
        document.getElementById("pair").innerHTML = ""

        let response = await fetch("http://127.0.0.1:8000/ask",{
            method:"POST",
            headers:{ "Content-Type":"application/json" },
            body: JSON.stringify({ question })
        })

        let data = await response.json()
        console.log("RESPONSE:", data)

        let answers = data.answers || []

        let answerA = answers[0] || "-"
        let answerB = answers[1] || "-"

        document.getElementById("answer").innerHTML = (data.combined_answer || "-").replace(/\n/g, "<br>")

        let entropyValue = data.entropies ? data.entropies[0] : null

        if (entropyValue !== null) {
            document.getElementById("entropy").innerText = entropyValue

            let riskText = ""
            let color = ""

            if (entropyValue < 1.5) {
                riskText = "🟢 LOW RISK (Jawaban cukup yakin)"
                color = "green"
            } else if (entropyValue < 2.5) {
                riskText = "🟠 MEDIUM RISK (Perlu verifikasi)"
                color = "orange"
            } else {
                riskText = "🔴 HIGH RISK (Potensi halusinasi)"
                color = "red"
            }

            document.getElementById("risk").innerText = riskText
            document.getElementById("risk").style.color = color
        } else {
            document.getElementById("entropy").innerText = "-"
            document.getElementById("risk").innerText = "-"
        }

        document.getElementById("pair").innerHTML = `
        <h3>Pilih jawaban terbaik</h3>

        <div>
        <p><b>A:</b> ${answerA}</p>
        <button onclick='choose(${JSON.stringify(question)}, ${JSON.stringify(answerA)}, ${JSON.stringify(answerB)})'>
        Pilih A
        </button>
        </div>

        <div>
        <p><b>B:</b> ${answerB}</p>
        <button onclick='choose(${JSON.stringify(question)}, ${JSON.stringify(answerB)}, ${JSON.stringify(answerA)})'>
        Pilih B
        </button>
        </div>
        `

    } catch (error) {
        console.error("ERROR:", error)
        alert("Terjadi error saat mengambil jawaban")
    }

    isLoading = false
}

async function choose(question, chosen, rejected){

await fetch("http://127.0.0.1:8000/save_preference_pair",{
method:"POST",
headers:{ "Content-Type":"application/json" },
body: JSON.stringify({
question,
chosen,
rejected
})
})

alert("Preference saved!")
}