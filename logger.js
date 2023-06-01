const { format, createLogger, transports } = require("winston")
const { combine, timestamp, label, printf } = format;

const timezoned = () => {
  return new Date().toLocaleString('en-US', {
      timeZone: 'Canada/Eastern'
  });
}

//Using the printf format.
const customFormat = printf(({ level, message, label, timestamp }) => {
  if(typeof message === "object"){
    return `${timestamp} [${label}] ${level}: ${JSON.stringify(message)}`;
  }
  else{
    return `${timestamp} [${label}] ${level}: ${message}`;
  }
});

const logger = function(fileName){
    return createLogger({
        level: "debug",
        format: combine(label({ label: fileName }), timestamp({format : timezoned}), customFormat),
        transports: [
            new transports.File({
              filename: "logs/standard.log",
            }),
            new transports.File({
              level: "error",
              filename: "logs/error.log",
            }),
            new transports.Console(),
        ],
    });
}

module.exports = {logger}