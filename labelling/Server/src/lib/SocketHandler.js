import http from "http"
import socServer from "socket.io"
import socClient from "socket.io-client"
const logger = require("../lib/Logger")(__filename)

class SocketHandler {
	constructor() {
		this.io = null
		this.server = null
		this.client = null
	}
	init(port) {
		this.server = http.createServer()
		this.io = socServer(this.server, {
			pingInterval: 40000,
			pingTimeout: 25000,
			upgradeTimeout: 21000, // default value is 10000ms, try changing it to 20k or more
			reconnectionDelay: 1000,
			reconnectionDelayMax: 5000
		})
		// this.io = socServer(this.server)

		this.io.on("connection", client => {
			client.on("init", function(data) {
				client.join(String(data.ROOM_ID))
				logger.info(`Join Socket In [ ${data.ROOM_ID} ]`)
				// client.emit("welcome", `hello! ${data.name}`)
			})
			client.on("event", data => {
				client.to(String(data.ROOM_ID)).emit("result", data)
			})
			client.on("disconnect", () => {
				/* … */
			})
		})
		this.server.listen(port)
		this.client = socClient.connect("http://localhost:3000")
	}

	close() {
		// console.log("close Server")
		this.server.close()
	}

	setSocData(data, type) {
		// logger.debug(JSON.stringify(data, null, 1))
		if (type === "REALTIME") data.ROOM_ID = data.IS_CD
		if (type === "TRAIN") data.ROOM_ID = data.AI_CD
		this.client.emit("event", data)
	}
}

module.exports = new SocketHandler()
