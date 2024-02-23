const { Tool } = require('langchain/tools');

class MusicPlayer extends Tool {
  constructor() {
    super();

    this.name = 'MusicPlayer';

    this.description =
      'When you are asked to start a session or an experience, use MusicPlayer to play some music. There are no arguments for MusicPlayer, as the output is a fixed audio file.';

    this.errorResponse = 'Sorry, I could not play music.';
  }

  async _call(input) {
    return '<audio>/music/ketamine.mp3</audio>';
  }
}

module.exports = MusicPlayer;
