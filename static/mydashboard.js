console.log("bonjour");

$.ajax({
	url: "/api/meteo",
	success: display_meteo
});

console.log("Au revoir");

function display_meteo(result){
	console.log("Résultat de la requête :", result);
	news_data = result["data"];
}