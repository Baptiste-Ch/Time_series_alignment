function incrementCounter() {
    var counter = parseInt(document.getElementById("counter").innerHTML);
    if (counter < {{ max_count }}) {
        counter++;
        document.getElementById("counter").innerHTML = counter;
        fetch('/increment');
        if (counter >= {{ max_count }}) {
            document.getElementById("increment-button").disabled = true;
        }
        if (counter > {{ min_count }}) {
            document.getElementById("decrease-button").disabled = false;
        }
    }
}
window.onload = function() {
    var counter = sessionStorage.getItem("counter");
    if (counter) {
        document.getElementById("counter").innerHTML = counter;
        if (counter >= {{ max_count }}) {
            document.getElementById("increment-button").disabled = true;
        }
    }
};
window.onbeforeunload = function() {
    sessionStorage.setItem("counter", "0");
};

function decreaseCounter() {
    var counter = parseInt(document.getElementById("counter").innerHTML);
    if (counter > {{ min_count }}) {
        counter--;
        document.getElementById("counter").innerHTML = counter;
        fetch('/decrease');
        if (counter <= {{ min_count }}) {
            document.getElementById("decrease-button").disabled = true;
        }
        if (counter < {{ max_count }}) {
            document.getElementById("increment-button").disabled = false;
        }
    }
}
window.onload = function() {
    var counter = sessionStorage.getItem("counter");
    if (counter) {
        document.getElementById("counter").innerHTML = counter;
        if (counter >= {{ max_count }}) {
            document.getElementById("decrease-button").disabled = true;
        }
    }
};
window.onbeforeunload = function() {
    sessionStorage.setItem("counter", "0");
};

function showDiv(event) {
    var x = document.getElementById("algorithm");
    if (x.style.display === "none") {
        x.style.display = "flex";
    } else {
        x.style.display = "none";
    }
    return false;
}