package edu.vit.pathsense;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.KeyEvent;
import android.webkit.GeolocationPermissions;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;

/**
 * PathSense for Android: the PathSense website, served by the team's laptop (start_pathsense.bat) on the same Wi-Fi,
 * in a full-screen app. The phone's own GPS answers "Use my location" (browsers only allow that over https), and the
 * Android file picker handles walk uploads and path photos.
 */
public class MainActivity extends Activity {
    private static final String PREFS = "pathsense", KEY_SERVER = "server";
    private static final int FILE_REQ = 1, LOC_REQ = 2;
    private WebView web;
    private ValueCallback<Uri[]> fileCallback;
    private boolean locating = false;
    private boolean loaded = false;
    private final Handler ui = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        web = new WebView(this);
        setContentView(web);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setGeolocationEnabled(true);
        s.setUserAgentString(s.getUserAgentString() + " PathSenseApp/1.0");
        web.addJavascriptInterface(new Bridge(), "PathSenseApp");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView v, String url) {
                if (server() != null && url.startsWith(server())) loaded = true;
            }

            @Override
            public void onReceivedError(WebView v, WebResourceRequest req, WebResourceError err) {
                if (req.isForMainFrame()) unreachable();
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest req) {
                Uri u = req.getUrl();
                if (server() != null && u.toString().startsWith(server())) return false;
                startActivity(new Intent(Intent.ACTION_VIEW, u));            // e.g. the GitHub link: open in the browser
                return true;
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView w, ValueCallback<Uri[]> cb, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = cb;
                try {
                    startActivityForResult(params.createIntent(), FILE_REQ);
                } catch (Exception e) {
                    fileCallback = null;
                    return false;
                }
                return true;
            }

            @Override
            public void onGeolocationPermissionsShowPrompt(String origin, GeolocationPermissions.Callback cb) {
                cb.invoke(origin, true, false);
            }
        });
        if (server() == null) askServer(null);
        else open(server());
    }

    /** Load the PathSense site; if it has not answered within 12 s, say so instead of showing a blank page. */
    private void open(String url) {
        loaded = false;
        web.loadDataWithBaseURL(null, "<html><body style='font-family:sans-serif;padding:32px;color:#0F1E3D'>"
                + "<h2>PathSense</h2><p>Connecting to " + url + " &hellip;</p></body></html>", "text/html", "utf-8", null);
        web.loadUrl(url);
        ui.removeCallbacksAndMessages(null);
        ui.postDelayed(() -> { if (!loaded) { web.stopLoading(); unreachable(); } }, 12_000);
    }

    private void unreachable() {
        if (loaded && web.getUrl() != null && web.getUrl().startsWith(server())) return;
        loaded = true;                                                    // one dialog per attempt
        askServer("Could not reach PathSense at " + server() + ".\n\nThe app shows PathSense from the team's laptop: "
                + "start_pathsense.bat must be running, and this phone must be on the same Wi-Fi or hotspot as the "
                + "laptop, or use the public https address if the laptop shares one.");
    }

    private String server() {
        return getSharedPreferences(PREFS, MODE_PRIVATE).getString(KEY_SERVER, null);
    }

    private void askServer(String message) {
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        input.setText(server() != null ? server() : "http://192.168.");
        input.setSelection(input.getText().length());
        new AlertDialog.Builder(this)
                .setTitle("PathSense server")
                .setMessage(message != null ? message
                        : "Type the address the laptop shows under \"On phones (same Wi-Fi)\", e.g. http://192.168.1.10:5000")
                .setView(input)
                .setCancelable(false)
                .setPositiveButton("Open", (d, w) -> {
                    String url = input.getText().toString().trim();
                    if (!url.startsWith("http")) url = "http://" + url;
                    while (url.endsWith("/")) url = url.substring(0, url.length() - 1);
                    getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_SERVER, url).apply();
                    open(url);
                })
                .setNegativeButton("Exit", (d, w) -> finish())
                .show();
    }

    @Override
    protected void onActivityResult(int req, int result, Intent data) {
        if (req == FILE_REQ && fileCallback != null) {
            fileCallback.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(result, data));
            fileCallback = null;
        }
        super.onActivityResult(req, result, data);
    }

    @Override
    public boolean onKeyDown(int code, KeyEvent ev) {
        if (code == KeyEvent.KEYCODE_BACK) {
            if (web.canGoBack()) {
                web.goBack();
            } else {
                new AlertDialog.Builder(this).setTitle("PathSense")
                        .setItems(new String[]{"Change server address", "Exit"}, (d, i) -> {
                            if (i == 0) askServer(null); else finish();
                        }).show();
            }
            return true;
        }
        return super.onKeyDown(code, ev);
    }

    // ---------------------------------------------------------------- native location for "Use my location"
    private class Bridge {
        @JavascriptInterface
        public void locate() { runOnUiThread(MainActivity.this::locateNow); }

        @JavascriptInterface
        public void changeServer() { runOnUiThread(() -> askServer(null)); }
    }

    private void locateNow() {
        if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION}, LOC_REQ);
            return;
        }
        LocationManager lm = (LocationManager) getSystemService(LOCATION_SERVICE);
        Location best = null;
        for (String p : new String[]{LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER}) {
            try {
                Location l = lm.getLastKnownLocation(p);
                if (l != null && (best == null || l.getTime() > best.getTime())) best = l;
            } catch (Exception ignored) { }
        }
        if (best != null && System.currentTimeMillis() - best.getTime() < 60_000) {
            send(best);
            return;
        }
        if (locating) return;
        locating = true;
        final Location fallback = best;
        final LocationListener once = new LocationListener() {
            @Override public void onLocationChanged(Location l) {
                if (!locating) return;
                locating = false; lm.removeUpdates(this); send(l);
            }
            @Override public void onStatusChanged(String p, int s, Bundle b) { }
            @Override public void onProviderEnabled(String p) { }
            @Override public void onProviderDisabled(String p) { }
        };
        try {
            if (lm.isProviderEnabled(LocationManager.GPS_PROVIDER))
                lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0, 0, once, Looper.getMainLooper());
            if (lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER))
                lm.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 0, 0, once, Looper.getMainLooper());
        } catch (Exception e) {
            locating = false;
            error("Location is switched off on this phone.");
            return;
        }
        new Handler(Looper.getMainLooper()).postDelayed(() -> {          // no fix within 15 s
            if (!locating) return;
            locating = false; lm.removeUpdates(once);
            if (fallback != null) send(fallback); else error("No GPS fix yet. Step outside, or pick the start on the map.");
        }, 15_000);
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] res) {
        if (req == LOC_REQ) {
            if (res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED) locateNow();
            else error("Location permission was not given. Pick the start on the map instead.");
        }
    }

    private void send(Location l) {
        web.evaluateJavascript("window.__psLocation && window.__psLocation(" + l.getLatitude() + "," + l.getLongitude() + ")", null);
    }

    private void error(String msg) {
        web.evaluateJavascript("window.__psLocationError && window.__psLocationError(" + org.json.JSONObject.quote(msg) + ")", null);
    }
}
