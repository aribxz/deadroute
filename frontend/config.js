(() => {
  const localHosts = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  const isSeparateLocalFrontend = localHosts && ["8000", "5500"].includes(window.location.port);
  window.DEADROUTE_API_BASE = window.DEADROUTE_API_BASE || (
    isSeparateLocalFrontend
      ? `${window.location.protocol}//${window.location.hostname}:5000`
      : window.location.origin
  );
})();
